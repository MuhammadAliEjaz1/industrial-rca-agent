"""
Phase 1: Data Setup for Industrial Root-Cause Investigation Agent

WHAT'S REAL vs SYNTHETIC (documented per project requirements):
- Sensor VALUES (temperature, torque, rotational speed, tool wear): REAL, from AI4I2020
- Failure TYPES and WHICH rows failed: REAL, from AI4I2020
- Machine IDs (grouping rows into "machines"): SYNTHETIC — constructed by us
- Timestamps: SYNTHETIC — constructed by us
- Maintenance records: FULLY SYNTHETIC — authored by us, not derived from data

KNOWN DATA QUIRKS (verified against the raw dataset, documented not hidden):
- 9 rows have machine_failure=1 but ALL 5 failure flags = 0. This is a genuine
  inconsistency in the raw AI4I2020 data itself, not an extraction bug. These
  are labeled "Unknown".
- 24 rows have 2+ simultaneous failure flags (23 with 2, 1 with 3). These are
  combined into a single compound label, e.g. "Heat Dissipation Failure + Power
  Failure", rather than silently keeping only the first flag (an earlier bug in
  this script that discarded co-occurring causes).
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from ucimlrepo import fetch_ucirepo

RANDOM_SEED = 42
NUM_MACHINES = 40
DB_PATH = "industrial_rca.db"

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# STEP 1: Fetch the real dataset
# ─────────────────────────────────────────────
def fetch_dataset() -> pd.DataFrame:
    print("Fetching AI4I 2020 dataset from UCI...")
    dataset = fetch_ucirepo(id=601)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)

    df.columns = [
        c.strip().lower().replace(" ", "_").replace("[", "").replace("]", "")
        for c in df.columns
    ]
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# STEP 2: Assign rows to synthetic machines (stratified by Type)
# ─────────────────────────────────────────────
def assign_machines(df: pd.DataFrame, num_machines: int):
    df = df.copy()
    df["machine_id"] = None

    machine_counter = 1
    machine_type_map = {}

    for product_type in df["type"].unique():
        subset_idx = df[df["type"] == product_type].index
        n_machines_this_type = max(1, int(num_machines * len(subset_idx) / len(df)))

        machine_ids_this_type = [f"M{machine_counter + i:03d}" for i in range(n_machines_this_type)]
        machine_counter += n_machines_this_type

        for mid in machine_ids_this_type:
            machine_type_map[mid] = product_type

        assigned = np.random.choice(machine_ids_this_type, size=len(subset_idx))
        df.loc[subset_idx, "machine_id"] = assigned

    print(f"Created {machine_counter - 1} synthetic machines across types {list(df['type'].unique())}")
    return df, machine_type_map


# ─────────────────────────────────────────────
# STEP 3: Assign synthetic timestamps (sorted per machine)
# ─────────────────────────────────────────────
def assign_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    end_date = datetime(2026, 9, 1)
    start_date = end_date - timedelta(days=548)

    df["timestamp"] = None
    for mid, group in df.groupby("machine_id"):
        n = len(group)
        offsets = np.sort(np.random.uniform(0, (end_date - start_date).total_seconds(), n))
        timestamps = [start_date + timedelta(seconds=float(o)) for o in offsets]
        df.loc[group.index, "timestamp"] = timestamps

    return df


# ─────────────────────────────────────────────
# STEP 4: Build maintenance records (fully synthetic/authored)
# ─────────────────────────────────────────────
MAINTENANCE_TEMPLATES = [
    ("Scheduled", "Routine inspection and lubrication of bearing assembly.", "T. Ahmed"),
    ("Scheduled", "Quarterly calibration check on torque and speed sensors.", "S. Malik"),
    ("Scheduled", "Coolant system inspection and top-up.", "T. Ahmed"),
    ("Reactive", "Replaced worn cutting tool after operator flagged excessive vibration.", "R. Hussain"),
    ("Reactive", "Investigated abnormal temperature reading; cleaned cooling fan.", "S. Malik"),
    ("Reactive", "Tightened loose motor mounting bolts.", "R. Hussain"),
]

def generate_maintenance_records(machine_type_map: dict) -> pd.DataFrame:
    records = []
    record_id = 1
    end_date = datetime(2026, 9, 1)

    for machine_id in machine_type_map:
        n_events = np.random.randint(3, 9)
        for _ in range(n_events):
            days_ago = np.random.randint(1, 548)
            date = end_date - timedelta(days=int(days_ago))
            maint_type, description, technician = MAINTENANCE_TEMPLATES[
                np.random.randint(0, len(MAINTENANCE_TEMPLATES))
            ]
            records.append({
                "record_id": record_id,
                "machine_id": machine_id,
                "date": date,
                "maintenance_type": maint_type,
                "description": description,
                "technician": technician,
                "parts_replaced": "Cutting tool" if "tool" in description.lower() else None,
            })
            record_id += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# STEP 5: Extract failure incidents (REAL, from actual failure rows)
# FIXED: combines multi-flag rows instead of keeping only the first flag
# ─────────────────────────────────────────────
FAILURE_LABELS = {
    "twf": "Tool Wear Failure",
    "hdf": "Heat Dissipation Failure",
    "pwf": "Power Failure",
    "osf": "Overstrain Failure",
    "rnf": "Random Failure",
}

def extract_failure_incidents(df: pd.DataFrame) -> pd.DataFrame:
    failures = df[df["machine_failure"] == 1].copy()
    incidents = []
    incident_id = 1

    combo_count = 0
    unknown_count = 0

    for _, row in failures.iterrows():
        active_modes = [FAILURE_LABELS[col] for col in FAILURE_LABELS if row.get(col, 0) == 1]

        if len(active_modes) == 0:
            failure_type = "Unknown"
            unknown_count += 1
        elif len(active_modes) == 1:
            failure_type = active_modes[0]
        else:
            failure_type = " + ".join(active_modes)  # e.g. "Heat Dissipation Failure + Power Failure"
            combo_count += 1

        incidents.append({
            "incident_id": incident_id,
            "machine_id": row["machine_id"],
            "date": row["timestamp"],
            "failure_type": failure_type,
            "description": f"Machine failure detected: {failure_type}. "
                            f"Air temp {row['air_temperature']:.1f}K, "
                            f"process temp {row['process_temperature']:.1f}K, "
                            f"speed {row['rotational_speed']:.0f}rpm, "
                            f"torque {row['torque']:.1f}Nm, "
                            f"tool wear {row['tool_wear']:.0f}min.",
            "resolution": None,
            "root_cause": failure_type,
        })
        incident_id += 1

    print(f"Failure incidents: {len(incidents)} total, {combo_count} multi-cause, {unknown_count} unknown (no flags set)")
    return pd.DataFrame(incidents)


# ─────────────────────────────────────────────
# STEP 6: Write everything to SQLite
# ─────────────────────────────────────────────
def build_database(df: pd.DataFrame, machine_type_map: dict, maintenance_df: pd.DataFrame, incidents_df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS machines;
    DROP TABLE IF EXISTS sensor_readings;
    DROP TABLE IF EXISTS maintenance_records;
    DROP TABLE IF EXISTS failure_incidents;

    CREATE TABLE machines (
        machine_id TEXT PRIMARY KEY,
        machine_type TEXT,
        install_date TEXT,
        location TEXT,
        status TEXT
    );

    CREATE TABLE sensor_readings (
        reading_id INTEGER PRIMARY KEY,
        machine_id TEXT,
        timestamp TEXT,
        air_temperature REAL,
        process_temperature REAL,
        rotational_speed REAL,
        torque REAL,
        tool_wear REAL,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
    );

    CREATE TABLE maintenance_records (
        record_id INTEGER PRIMARY KEY,
        machine_id TEXT,
        date TEXT,
        maintenance_type TEXT,
        description TEXT,
        technician TEXT,
        parts_replaced TEXT,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
    );

    CREATE TABLE failure_incidents (
        incident_id INTEGER PRIMARY KEY,
        machine_id TEXT,
        date TEXT,
        failure_type TEXT,
        description TEXT,
        resolution TEXT,
        root_cause TEXT,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
    );
    """)

    locations = ["Line A - Bay 1", "Line A - Bay 2", "Line B - Bay 1", "Line B - Bay 2", "Line C - Bay 1"]

    for mid, mtype in machine_type_map.items():
        install_date = datetime(2026, 9, 1) - timedelta(days=int(np.random.randint(365, 1825)))
        cur.execute(
            "INSERT INTO machines VALUES (?, ?, ?, ?, ?)",
            (mid, mtype, install_date.strftime("%Y-%m-%d"), np.random.choice(locations), "operational")
        )

    reading_id = 1
    for _, row in df.iterrows():
        cur.execute(
            "INSERT INTO sensor_readings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (reading_id, row["machine_id"], str(row["timestamp"]),
             row["air_temperature"], row["process_temperature"],
             row["rotational_speed"], row["torque"], row["tool_wear"])
        )
        reading_id += 1

    for _, row in maintenance_df.iterrows():
        cur.execute(
            "INSERT INTO maintenance_records VALUES (?, ?, ?, ?, ?, ?, ?)",
            (row["record_id"], row["machine_id"], str(row["date"]),
             row["maintenance_type"], row["description"], row["technician"], row["parts_replaced"])
        )

    for _, row in incidents_df.iterrows():
        cur.execute(
            "INSERT INTO failure_incidents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (row["incident_id"], row["machine_id"], str(row["date"]),
             row["failure_type"], row["description"], row["resolution"], row["root_cause"])
        )

    conn.commit()
    conn.close()
    print(f"Database written to {DB_PATH}")


# ─────────────────────────────────────────────
# STEP 7: Verification queries
# ─────────────────────────────────────────────
def verify(machine_type_map: dict):
    conn = sqlite3.connect(DB_PATH)

    print("\n=== VERIFICATION ===")
    print("Machines:", conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0])
    print("Sensor readings:", conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0])
    print("Maintenance records:", conn.execute("SELECT COUNT(*) FROM maintenance_records").fetchone()[0])
    print("Failure incidents:", conn.execute("SELECT COUNT(*) FROM failure_incidents").fetchone()[0])

    print("\nFailure type breakdown (should now include combined labels):")
    for row in conn.execute("SELECT failure_type, COUNT(*) FROM failure_incidents GROUP BY failure_type ORDER BY COUNT(*) DESC").fetchall():
        print(" ", row)

    conn.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_dataset()
    df, machine_type_map = assign_machines(df, NUM_MACHINES)
    df = assign_timestamps(df)
    maintenance_df = generate_maintenance_records(machine_type_map)
    incidents_df = extract_failure_incidents(df)

    build_database(df, machine_type_map, maintenance_df, incidents_df)
    verify(machine_type_map)