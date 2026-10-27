"""
Phase 8, step 3: run the agent on the evaluation cases.   *** THIS CALLS GROQ ***

Run from backend/:
    python evaluation/08_run_agent.py --dev-only         # the 12 development (pilot) cases -> agent_results_dev.jsonl
    python evaluation/08_run_agent.py                    # the TEST cases: held-out incidents + healthy cases, minus the dev cases
    python evaluation/08_run_agent.py --limit 10         # only the first N cases (spread over all failure types)
    python evaluation/08_run_agent.py --all-incidents    # also the incidents the classifier trained on
    python evaluation/08_run_agent.py --no-maintenance   # ablation: hide maintenance records
    python evaluation/08_run_agent.py --tag NAME         # write to agent_results_NAME.jsonl

The 12 pilot cases were used to find bugs and tune the agent, so they are a DEV set. The headline results must be
reported on the other cases only; the default run excludes the dev cases (use --include-dev to override).

Results are appended line by line to evaluation/agent_results.jsonl (one JSON object per case), so an
interrupted run loses nothing; running it again skips finished cases and retries failed ones.
Every full report is saved to evaluation/reports/<case_id>.json for reading by hand.

The agent never sees the stored incident description (it contains the answer). It gets a neutral sentence,
the machine, the time of the incident (as_of), and the incident_id so its own row is excluded from search.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import BACKEND, score_case, results_suffix
from graph.checks import is_null_hypothesis

EVAL_DIR = BACKEND / "evaluation"
CASES_FILE = EVAL_DIR / "eval_cases.json"
ROUTINE = "Routine automated check flagged this machine for investigation."
# The cases of the first pilot run. They were looked at and the agent was changed because of them.
DEV_CASE_IDS = {"inc-117", "inc-156", "inc-139", "inc-23", "inc-14", "inc-317", "inc-105", "inc-87", "inc-204",
                "healthy-14", "healthywear-2853", "inc-123"}


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
        "advisory": bool(report.get("advisories")),
        "unverified_facts": sum("[UNVERIFIED" in f for f in facts),
        "ml_class": (report.get("ml_analysis") or {}).get("predicted_class"),
        "n_recent_readings": len(result.get("sensor_readings", [])),
        "n_history_readings": (result.get("sensor_statistics") or {}).get("reading_count"),
    }


def invoke_with_usage(graph, state: dict):
    """Runs the graph and returns (result, total_tokens). Tokens are None if the counter is unavailable."""
    try:
        from langchain_core.callbacks import get_usage_metadata_callback
    except Exception:  # noqa: BLE001
        return graph.invoke(state), None
    with get_usage_metadata_callback() as cb:
        result = graph.invoke(state)
    usage = getattr(cb, "usage_metadata", None) or {}
    total = sum(int(v.get("total_tokens", 0)) for v in usage.values())
    return result, (total or None)


def is_daily_limit(text: str) -> bool:
    return "tokens per day" in text.lower() or "(tpd)" in text.lower()


def retry_hint(text: str) -> str:
    """Groq's own 'try again in ...' wording, e.g. '7m20.208s' (not capped)."""
    m = re.search(r"try again in ([0-9hms.]+)", text)
    return m.group(1).rstrip(".") if m else "an unknown time"


def retry_after_seconds(text: str, default: float = 65.0) -> float:
    m = re.search(r"try again in (?:(\d+)m)?\s*(?:([\d.]+)s)?", text)
    if m and (m.group(1) or m.group(2)):
        return min(float(m.group(1) or 0) * 60 + float(m.group(2) or 0) + 2, 180.0)
    return default


def run(cases: list, graph, out_path: Path, reports_dir: Path, pause: float = 1.0) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"{len(cases)} selected, {len(cases) - len(todo)} already done, {len(todo)} to run")
    stats = {"ok": 0, "error": 0, "stopped_daily_limit": False}
    for n, case in enumerate(todo, 1):
        start = time.time()
        row = None
        for attempt in (1, 2):
            try:
                result, tokens = invoke_with_usage(graph, case_state(case))
                row = make_row(case, result, time.time() - start)
                row["tokens"] = tokens
                (reports_dir / f"{case['case_id']}.json").write_text(
                    json.dumps(result["final_report"], indent=1, default=str), encoding="utf-8")
                break
            except KeyboardInterrupt:
                print("\nInterrupted. Finished cases are saved; run again to continue.")
                return stats
            except Exception as e:  # noqa: BLE001 - record and carry on with the next case
                text = f"{type(e).__name__}: {e}"
                if is_daily_limit(text):
                    print(f"\nGroq's DAILY token limit is used up. Groq says: try again in {retry_hint(text)} "
                          f"(that is for a request of this size; the quota frees up gradually).\n"
                          f"Stopping now; every finished case is saved. Run the same command again later.")
                    stats["stopped_daily_limit"] = True
                    return stats
                if attempt == 1 and ("429" in text or "rate limit" in text.lower()):
                    wait = retry_after_seconds(text)
                    print(f"  per-minute rate limit on {case['case_id']}, waiting {wait:.0f} s and retrying once")
                    time.sleep(wait)
                    continue
                row = {"case_id": case["case_id"], "kind": case["kind"], "truth_raw": case["truth_raw"],
                       "status": "error", "error": text[:300], "elapsed_s": round(time.time() - start, 1)}
                break
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        stats[row["status"]] += 1
        tag = (f"{row['score']:9s} final={row['final_cause']} ({row['final_confidence']})"
               if row["status"] == "ok" else f"ERROR {row['error'][:80]}")
        tok = f"  {row['tokens']} tokens" if row.get("tokens") else ""
        print(f"[{n}/{len(todo)}] {case['case_id']:18s} truth={case['truth_raw']:34s} {tag}  {row['elapsed_s']}s{tok}")
        time.sleep(pause)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases (spread over all types)")
    ap.add_argument("--all-incidents", action="store_true", help="include incidents the classifier trained on")
    ap.add_argument("--no-maintenance", action="store_true", help="ablation: hide all maintenance records")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds to wait between cases")
    ap.add_argument("--dev-only", action="store_true", help="run only the 12 development (pilot) cases")
    ap.add_argument("--include-dev", action="store_true", help="do not exclude the development cases")
    ap.add_argument("--tag", default=None, help="name for the results file: agent_results_<tag>.jsonl")
    args = ap.parse_args()

    if not CASES_FILE.exists():
        raise SystemExit("Run evaluation/08_make_eval_sets.py first.")
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if not args.all_incidents:
        cases = [c for c in cases if c["in_test"]]
    if args.dev_only:
        cases = [c for c in cases if c["case_id"] in DEV_CASE_IDS]
    elif not args.include_dev:
        cases = [c for c in cases if c["case_id"] not in DEV_CASE_IDS]
    cases = order_cases(cases)
    if args.limit:
        cases = cases[:args.limit]

    import graph.nodes as nodes
    from graph.build_graph import build_graph
    if args.no_maintenance:
        class _NoMaintenance:
            def invoke(self, a):
                return {"records": [], "record_count": 0}
        nodes.get_maintenance_history = _NoMaintenance()
    suffix = results_suffix(args.tag, args.dev_only, args.no_maintenance)
    out_path = EVAL_DIR / f"agent_results{suffix}.jsonl"
    reports_dir = EVAL_DIR / f"reports{suffix}"
    print(f"Writing to {out_path}")
    stats = run(cases, build_graph(), out_path, reports_dir, pause=args.pause)
    print(f"\nFinished: {stats['ok']} ok, {stats['error']} errors. Next: python evaluation/08_report.py")


if __name__ == "__main__":
    main()
