"""
Checks that the 'as_of' time-travel option really hides the future. Runs against your REAL database and the
real classifier. No Groq, no network. From backend/:   python test_as_of.py
"""
import sqlite3, sys
sys.path.insert(0, ".")

from database.queries import DB_PATH
from tools.investigation_tools import (
    get_sensor_data, get_maintenance_history, get_failure_history,
    run_anomaly_detection, calculate_sensor_statistics,
)

c = sqlite3.connect(DB_PATH)
# an incident whose machine has at least 5 later readings, so a leak would be visible
iid, mid, date = c.execute("""
    SELECT fi.incident_id, fi.machine_id, fi.date FROM failure_incidents fi
    WHERE (SELECT COUNT(*) FROM sensor_readings sr WHERE sr.machine_id = fi.machine_id AND sr.timestamp > fi.date) >= 5
    ORDER BY fi.incident_id LIMIT 1""").fetchone()
print(f"Using incident {iid} on {mid} at {date}")

# 1. recent readings: the incident's own reading is the newest, nothing later is visible
r = get_sensor_data.invoke({"machine_id": mid, "limit": 20, "as_of": date})
assert r["readings"][0]["timestamp"] == date, r["readings"][0]
assert all(x["timestamp"] <= date for x in r["readings"])
assert get_sensor_data.invoke({"machine_id": mid, "limit": 1})["readings"][0]["timestamp"] > date  # without as_of: later ones exist
print("OK  readings: newest visible reading is the incident's own; later ones hidden")

# 2. the classifier is run on THAT reading, not the newest one
a = run_anomaly_detection.invoke({"machine_id": mid, "as_of": date})
assert a["found"] and a["based_on_reading_timestamp"] == date, a
print("OK  classifier input is the incident's reading ->", a["predicted_class"], f"({a['confidence']:.3f})")

# 3. statistics stop at the incident: fewer readings, and 'latest' is the incident's own tool wear
s_then = calculate_sensor_statistics.invoke({"machine_id": mid, "as_of": date})
s_now = calculate_sensor_statistics.invoke({"machine_id": mid})
own = c.execute("SELECT tool_wear FROM sensor_readings WHERE machine_id=? AND timestamp=?", (mid, date)).fetchone()[0]
assert s_then["reading_count"] < s_now["reading_count"]
assert abs(s_then["channels"]["tool_wear"]["latest"] - own) < 0.001
print(f"OK  statistics: {s_then['reading_count']} readings up to the incident vs {s_now['reading_count']} in total")

# 4. this machine's incident history: strictly earlier only, never the incident itself
h = get_failure_history.invoke({"machine_id": mid, "exclude_incident_id": iid, "as_of": date})
assert all(x["date"] < date and x["incident_id"] != iid for x in h["incidents"])
h_all = get_failure_history.invoke({"machine_id": mid, "exclude_incident_id": iid})
print(f"OK  failure history: {h['incident_count']} earlier incidents (without as_of: {h_all['incident_count']})")

# 5. maintenance: nothing dated after the incident
m = get_maintenance_history.invoke({"machine_id": mid, "as_of": date})
assert all(x["date"] <= date for x in m["records"])
m_all = get_maintenance_history.invoke({"machine_id": mid})
print(f"OK  maintenance: {m['record_count']} records up to the incident (without as_of: {m_all['record_count']})")

print("\nALL as_of TESTS PASSED")
