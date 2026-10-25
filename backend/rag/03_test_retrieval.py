"""
Phase 3 check: does retrieval return the right document for realistic questions?
Run from the backend/ folder:
    python rag/03_test_retrieval.py

This is a small starter for the Phase 8 RAG evaluation, not a full benchmark.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import search_technical_documents, search_similar_incidents

# (query, set of acceptable source files for a correct top result)
TEST_CASES = [
    ("process temperature is barely above air temperature and the spindle is running slowly",
     {"heat_dissipation_failure.md"}),
    ("tool wear times torque exceeds the limit for quality M",
     {"overstrain_failure.md"}),
    ("derived power dropped below 3500 watts",
     {"power_failure.md"}),
    ("every sensor is normal and no limit is violated but the machine stopped",
     {"random_failure.md"}),
    ("the tool has 220 minutes of wear, how do I inspect the cutting edge",
     {"tool_wear_failure.md"}),
    ("torque and power are both too high at the same time",
     {"compound_failures.md", "power_failure.md", "overstrain_failure.md"}),
    ("how often should torque and speed sensors be calibrated",
     {"preventive_maintenance_schedule.md", "sensor_interpretation_guide.md"}),
    ("what are the steps of an investigation",
     {"general_troubleshooting_procedure.md"}),
]


def run_manual_tests():
    top1 = top3 = 0
    for query, acceptable in TEST_CASES:
        results = search_technical_documents(query, k=3)
        sources = [r["source"] for r in results]
        hit1 = sources[0] in acceptable
        hit3 = any(s in acceptable for s in sources)
        top1 += hit1
        top3 += hit3
        print(f"\nQ: {query}")
        for r in results:
            print(f"   {r['similarity']:.3f}  {r['source']} > {r['section']}")
        print(f"   top-1 {'PASS' if hit1 else 'FAIL'} | top-3 {'PASS' if hit3 else 'FAIL'}")
    n = len(TEST_CASES)
    print(f"\n=== Manual retrieval: top-1 {top1}/{n}, top-3 {top3}/{n} ===")


def run_incident_test():
    query = "Heat dissipation failure with a small temperature gap and low rotational speed"
    results = search_similar_incidents(query, k=5)
    print(f"\n=== Incident search: {query!r} ===")
    for r in results:
        print(f"   {r['similarity']:.3f}  incident {r['incident_id']} ({r['machine_id']}) - {r['failure_type']}")
    matches = sum("Heat Dissipation" in r["failure_type"] for r in results)
    print(f"   {matches}/5 results involve Heat Dissipation")


if __name__ == "__main__":
    run_manual_tests()
    run_incident_test()
