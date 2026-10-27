"""Read-only check of how failure_incidents relate to sensor_readings (Phase 8 prep). Prints counts only."""
import re, sqlite3

c = sqlite3.connect("industrial_rca.db")
incidents = c.execute("select incident_id, machine_id, date, failure_type, description, root_cause "
                      "from failure_incidents").fetchall()
pat = re.compile(r"Air temp ([\d.]+)K, process temp ([\d.]+)K, speed ([\d.]+)rpm, torque ([\d.]+)Nm, tool wear ([\d.]+)min")

parsed = linked = same_values = label_in_desc = rc_equals_type = 0
for iid, mid, date, ftype, desc, root in incidents:
    m = pat.search(desc or "")
    parsed += bool(m)
    label_in_desc += (ftype or "").lower() in (desc or "").lower()
    rc_equals_type += (root == ftype)
    row = c.execute("select air_temperature, process_temperature, rotational_speed, torque, tool_wear "
                    "from sensor_readings where machine_id=? and timestamp=?", (mid, date)).fetchone()
    if row:
        linked += 1
        same_values += bool(m) and all(abs(float(a) - b) < 0.051 for a, b in zip(m.groups(), row))

n = len(incidents)
print(f"incidents: {n}")
print(f"description values parsed: {parsed}/{n}")
print(f"a sensor_readings row exists with the same machine_id + timestamp as the incident date: {linked}/{n}")
print(f"... and its values match the numbers in the description: {same_values}/{n}")
print(f"failure type written inside the description text (answer leak): {label_in_desc}/{n}")
print(f"root_cause column identical to failure_type: {rc_equals_type}/{n}")
print()
for ftype, k in c.execute("select failure_type, count(*) from failure_incidents group by 1 order by 2 desc"):
    print(f"{k:4d}  {ftype}")
