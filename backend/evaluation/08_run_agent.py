"""
Phase 8, step 3: run the agent on the evaluation cases.   *** THIS CALLS GROQ ***

Run from backend/:
    python evaluation/08_run_agent.py --limit 12        # pilot: 12 cases spread over all failure types
    python evaluation/08_run_agent.py                    # all held-out incidents + healthy cases
    python evaluation/08_run_agent.py --all-incidents    # also the incidents the classifier trained on
    python evaluation/08_run_agent.py --no-maintenance   # ablation: hide maintenance records

Results are appended line by line to evaluation/agent_results.jsonl (one JSON object per case), so an
interrupted run loses nothing; running it again skips finished cases and retries failed ones.
Every full report is saved to evaluation/reports/<case_id>.json for reading by hand.

The agent never sees the stored incident description (it contains the answer). It gets a neutral sentence,
the machine, the time of the incident (as_of), and the incident_id so its own row is excluded from search.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import BACKEND, score_case
from graph.checks import is_null_hypothesis

EVAL_DIR = BACKEND / "evaluation"
CASES_FILE = EVAL_DIR / "eval_cases.json"
ROUTINE = "Routine automated check flagged this machine for investigation."


def case_state(case: dict) -> dict:
    """Graph input for one case. Never contains the failure label."""
    if case["kind"] == "incident":
        description = f"A failure was reported on this machine at {case['as_of']}. Investigate the root cause."
        failure_reported, incident_id = True, case["incident_id"]
    else:
        description, failure_reported, incident_id = ROUTINE, False, None
    return {"machine_id": case["machine_id"], "incident_description": description, "incident_id": incident_id,
            "as_of": case["as_of"], "failure_reported": failure_reported, "retry_count": 0}


def agent_components(report: dict) -> set:
    """Causes the agent names: every non-null candidate rated medium or high."""
    return {e["cause"] for e in report.get("candidate_root_causes", [])
            if not is_null_hypothesis(e["cause"]) and e["confidence"] in ("medium", "high")}


def order_cases(cases: list) -> list:
    """Round-robin over true labels, so a small --limit still covers every failure type."""
    groups = {}
    for c in sorted(cases, key=lambda c: c["case_id"]):
        groups.setdefault(c["kind"] if c["kind"] != "incident" else c["truth_raw"], []).append(c)
    ordered, i = [], 0
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].pop(0))
        i += 1
    return ordered


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    done.add(row["case_id"])
    return done


def make_row(case: dict, result: dict, elapsed: float) -> dict:
    report = result["final_report"]
    predicted = agent_components(report)
    facts = report.get("observed_facts", [])
    candidates = report.get("candidate_root_causes", [])
    return {
        "case_id": case["case_id"], "kind": case["kind"], "truth_raw": case["truth_raw"],
        "expected": case["expected"], "status": "ok", "elapsed_s": round(elapsed, 1),
        "mode": report.get("mode"), "final_cause": report.get("most_supported_cause"),
        "final_confidence": report.get("confidence_level"), "predicted": sorted(predicted),
        "score": score_case(set(case["expected"]), predicted),
        "candidates": [{"cause": e["cause"], "confidence": e["confidence"],
                        "capped": "confidence_note" in e} for e in candidates],
        "retries": result.get("retry_count", 0),
        "unverified_facts": sum("[UNVERIFIED" in f for f in facts),
        "ml_class": (report.get("ml_analysis") or {}).get("predicted_class"),
        "n_recent_readings": len(result.get("sensor_readings", [])),
        "n_history_readings": (result.get("sensor_statistics") or {}).get("reading_count"),
    }


def run(cases: list, graph, out_path: Path, reports_dir: Path, pause: float = 1.0) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"{len(cases)} selected, {len(cases) - len(todo)} already done, {len(todo)} to run")
    stats = {"ok": 0, "error": 0}
    for n, case in enumerate(todo, 1):
        start = time.time()
        row = None
        for attempt in (1, 2):
            try:
                result = graph.invoke(case_state(case))
                row = make_row(case, result, time.time() - start)
                (reports_dir / f"{case['case_id']}.json").write_text(
                    json.dumps(result["final_report"], indent=1, default=str), encoding="utf-8")
                break
            except KeyboardInterrupt:
                print("\nInterrupted. Finished cases are saved; run again to continue.")
                return stats
            except Exception as e:  # noqa: BLE001 - record and carry on with the next case
                text = f"{type(e).__name__}: {e}"
                if attempt == 1 and ("429" in text or "rate limit" in text.lower()):
                    print(f"  rate limit hit on {case['case_id']}, waiting 65 s and retrying once")
                    time.sleep(65)
                    continue
                row = {"case_id": case["case_id"], "kind": case["kind"], "truth_raw": case["truth_raw"],
                       "status": "error", "error": text[:300], "elapsed_s": round(time.time() - start, 1)}
                break
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        stats[row["status"]] += 1
        tag = (f"{row['score']:9s} final={row['final_cause']} ({row['final_confidence']})"
               if row["status"] == "ok" else f"ERROR {row['error'][:80]}")
        print(f"[{n}/{len(todo)}] {case['case_id']:18s} truth={case['truth_raw']:34s} {tag}  {row['elapsed_s']}s")
        time.sleep(pause)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases (spread over all types)")
    ap.add_argument("--all-incidents", action="store_true", help="include incidents the classifier trained on")
    ap.add_argument("--no-maintenance", action="store_true", help="ablation: hide all maintenance records")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds to wait between cases")
    args = ap.parse_args()

    if not CASES_FILE.exists():
        raise SystemExit("Run evaluation/08_make_eval_sets.py first.")
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if not args.all_incidents:
        cases = [c for c in cases if c["in_test"]]
    cases = order_cases(cases)
    if args.limit:
        cases = cases[:args.limit]

    import graph.nodes as nodes
    from graph.build_graph import build_graph
    suffix = ""
    if args.no_maintenance:
        class _NoMaintenance:
            def invoke(self, a):
                return {"records": [], "record_count": 0}
        nodes.get_maintenance_history = _NoMaintenance()
        suffix = "_nomaint"
    out_path = EVAL_DIR / f"agent_results{suffix}.jsonl"
    reports_dir = EVAL_DIR / f"reports{suffix}"
    print(f"Writing to {out_path}")
    stats = run(cases, build_graph(), out_path, reports_dir, pause=args.pause)
    print(f"\nFinished: {stats['ok']} ok, {stats['error']} errors. Next: python evaluation/08_report.py")


if __name__ == "__main__":
    main()
