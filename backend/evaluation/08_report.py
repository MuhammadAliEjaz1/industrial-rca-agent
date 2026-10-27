"""
Phase 8, step 4: score the agent's results and compare them with the no-LLM baseline on the SAME cases.

Run from backend/:   python evaluation/08_report.py [--dev-only] [--tag NAME] [--no-maintenance]
(with no flags it reads agent_results.jsonl, the TEST run)
No LLM, no network.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import (load_split, truth_components, ml_components, rule_components, score_case, reading_of,
                               results_suffix, BACKEND)
from evaluation.report_lib import summarize, format_report


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    suffix = results_suffix(tag, "--dev-only" in sys.argv, "--no-maintenance" in sys.argv)
    path = BACKEND / "evaluation" / f"agent_results{suffix}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found - run evaluation/08_run_agent.py first.")
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["case_id"]] = row           # a later line (a retry) replaces an earlier one
    ok = [r for r in latest.values() if r["status"] == "ok"]
    errors = [r for r in latest.values() if r["status"] != "ok"]

    df, _ = load_split()
    by_incident = {int(r["incident_id"]): r for _, r in df[df["incident_id"].notna()].iterrows()}
    by_reading = {int(r["reading_id"]): r for _, r in df.iterrows()}
    base = {}
    for r in ok:
        cid = r["case_id"]
        row = by_incident[int(cid.split("-")[1])] if cid.startswith("inc-") else by_reading[int(cid.split("-")[1])]
        predicted = rule_components(reading_of(row), False) | ml_components(row["ml_pred"])
        base[cid] = score_case(truth_components(row["raw_label"]), predicted)

    print(format_report(summarize(ok, base), n_errors=len(errors)))
    for r in errors[:5]:
        print(f"  error: {r['case_id']}: {r.get('error')}")


if __name__ == "__main__":
    main()
