"""
Phase 5 check: runs a REAL investigation end to end against your actual database,
trained model, vector store, and Groq API key. This is the first time everything
from Phases 1-5 runs together as one pipeline.

Run from backend/:
    python graph/05_test_agent.py [machine_id] [--failure]
(without --failure it is a routine check with no failure reported -> health-check mode)
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import queries as db
from graph.build_graph import build_graph


def pick_machine_id() -> str:
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute("SELECT machine_id FROM machines LIMIT 1").fetchone()
    conn.close()
    if not row:
        raise SystemExit("No machines found — did you run Phase 1?")
    return row[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    machine_id = args[0] if args else pick_machine_id()
    failure_reported = "--failure" in sys.argv[1:]   # default: routine health check, nothing reported
    print(f"Running a full investigation for machine_id = {machine_id} "
          f"({'FAILURE reported' if failure_reported else 'routine check, no failure reported'})\n")

    graph = build_graph()
    result = graph.invoke({
        "machine_id": machine_id,
        "incident_description": ("The machine stopped unexpectedly; the operator requests a root-cause investigation."
                                 if failure_reported else
                                 "Routine automated check flagged this machine for investigation."),
        "failure_reported": failure_reported,
        "incident_id": None,
        "retry_count": 0,
    })

    print("=== Investigation timeline ===")
    for entry in result["investigation_log"]:
        print(f"  [{entry['step']:25s}] {entry['summary']}")

    print("\n=== Hypotheses generated ===")
    for h in result.get("hypotheses", []):
        print(f"  - {h['cause']}: {h['rationale']}")

    print("\n=== Hypothesis evaluations ===")
    for e in result.get("hypothesis_evaluations", []):
        print(f"\n  Cause: {e['cause']} (confidence: {e['confidence']})")
        print(f"    Supporting: {e['supporting_evidence']}")
        print(f"    Contradicting: {e['contradicting_evidence']}")
        print(f"    Reasoning: {e['reasoning']}")

    print(f"\n=== Root cause determination ===")
    rc = result.get("root_cause", {})
    print(f"  Most supported cause: {rc.get('most_supported_cause')}")
    print(f"  Confidence: {rc.get('confidence_level')}")
    print(f"  Justification: {rc.get('justification')}")
    print(f"  Observed facts: {rc.get('observed_facts')}")
    print(f"  ML predictions: {rc.get('ml_predictions')}")
    print(f"  Unsupported speculation: {rc.get('unsupported_speculation')}")

    print(f"\n=== retry_count used: {result.get('retry_count', 0)} ===")

    out_path = Path("last_investigation_report.json")
    out_path.write_text(json.dumps(result["final_report"], indent=2, default=str))
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
