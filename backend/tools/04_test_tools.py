"""
Phase 4 check: exercises all 7 tools against your REAL database, model, and vector
store. Run this after Phases 1-3 are all done and industrial_rca.db / the trained
model / the Chroma store all exist.

Run from backend/:
    python tools/04_test_tools.py <a_real_machine_id>

If you don't pass a machine_id, it picks the first one in your database.
"""

import sys
import pprint
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import queries as db
from tools.investigation_tools import (
    get_sensor_data, get_maintenance_history, get_failure_history,
    run_anomaly_detection, search_technical_documents,
    calculate_sensor_statistics, search_similar_incidents,
)


def pick_machine_id() -> str:
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute("SELECT machine_id FROM machines LIMIT 1").fetchone()
    conn.close()
    if not row:
        raise SystemExit("No machines found — did you run Phase 1 (database/01_setup_data.py)?")
    return row[0]


def main():
    machine_id = sys.argv[1] if len(sys.argv) > 1 else pick_machine_id()
    print(f"Testing against machine_id = {machine_id}\n")

    print("=== 1. get_sensor_data ===")
    pprint.pprint(get_sensor_data.invoke({"machine_id": machine_id, "limit": 3}))

    print("\n=== 2. get_maintenance_history ===")
    r = get_maintenance_history.invoke({"machine_id": machine_id})
    print(f"found={r['found']} record_count={r['record_count']}")

    print("\n=== 3. get_failure_history ===")
    r = get_failure_history.invoke({"machine_id": machine_id})
    print(f"found={r['found']} incident_count={r['incident_count']}")

    print("\n=== 4. run_anomaly_detection (needs Phase 2's trained model) ===")
    r = run_anomaly_detection.invoke({"machine_id": machine_id})
    pprint.pprint(r)

    print("\n=== 5. search_technical_documents (needs Phase 3's vector store) ===")
    r = search_technical_documents.invoke({"query": "machine running too slow with a narrow temperature gap"})
    for item in r["results"]:
        print(f"  {item['similarity']:.3f}  {item['source']} > {item['section']}")

    print("\n=== 6. calculate_sensor_statistics ===")
    r = calculate_sensor_statistics.invoke({"machine_id": machine_id})
    for ch, stats in r.get("channels", {}).items():
        print(f"  {ch:20s} trend={stats['trend']:10s} latest={stats['latest']}")

    print("\n=== 7. search_similar_incidents (sensor-distance path) ===")
    latest = get_sensor_data.invoke({"machine_id": machine_id, "limit": 1})
    if latest["found"] and latest["readings"]:
        reading = latest["readings"][0]
        r = search_similar_incidents.invoke({
            "description": "recent reading for this machine",
            "air_temperature": reading["air_temperature"],
            "process_temperature": reading["process_temperature"],
            "rotational_speed": reading["rotational_speed"],
            "torque": reading["torque"],
            "tool_wear": reading["tool_wear"],
            "k": 5,
        })
        print(f"ranked_by={r['ranked_by']}")
        for item in r["results"]:
            print(f"  dist={item['sensor_distance']}  incident {item['incident_id']} ({item['machine_id']}) - {item['failure_type']}")

    print("\nDone. Review each section above — especially #4 and #5, which couldn't be "
          "tested without your real model/vector store.")


if __name__ == "__main__":
    main()
