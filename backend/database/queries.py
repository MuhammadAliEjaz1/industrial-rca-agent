"""
Plain sqlite3 query helpers for the 4 tables. No ORM, consistent with Phases 1-2.
Every function returns a fixed shape (dict or list of dicts) even when nothing is
found, so tool code never has to guess whether a None means "no rows" or "error".
"""

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "industrial_rca.db"


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def machine_exists(machine_id: str, db_path: Path = DB_PATH) -> bool:
    conn = _connect(db_path)
    row = conn.execute("SELECT 1 FROM machines WHERE machine_id = ?", (machine_id,)).fetchone()
    conn.close()
    return row is not None


def get_recent_readings(machine_id: str, limit: int = 20, db_path: Path = DB_PATH,
                        as_of: Optional[str] = None) -> list[dict]:
    """Newest first. as_of (same text format as the timestamps) hides every reading AFTER that moment,
    so a historical incident can be investigated as if it were happening 'now'."""
    sql = ("SELECT timestamp, air_temperature, process_temperature, rotational_speed, torque, tool_wear "
           "FROM sensor_readings WHERE machine_id = ?")
    params: list = [machine_id]
    if as_of is not None:
        sql += " AND timestamp <= ?"
        params.append(as_of)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    conn = _connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_readings(machine_id: str, db_path: Path = DB_PATH, as_of: Optional[str] = None) -> list[dict]:
    """No limit — used for statistics, where we want the full history, not just recent.
    With as_of, the history stops at that moment (no look-ahead)."""
    sql = ("SELECT timestamp, air_temperature, process_temperature, rotational_speed, torque, tool_wear "
           "FROM sensor_readings WHERE machine_id = ?")
    params: list = [machine_id]
    if as_of is not None:
        sql += " AND timestamp <= ?"
        params.append(as_of)
    sql += " ORDER BY timestamp ASC"
    conn = _connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_maintenance_records(machine_id: str, db_path: Path = DB_PATH, as_of: Optional[str] = None) -> list[dict]:
    sql = ("SELECT date, maintenance_type, description, technician, parts_replaced "
           "FROM maintenance_records WHERE machine_id = ?")
    params: list = [machine_id]
    if as_of is not None:
        sql += " AND date <= ?"   # 'YYYY-MM-DD' sorts before 'YYYY-MM-DD hh:mm:ss', so same-day records are kept
        params.append(as_of)
    sql += " ORDER BY date DESC"
    conn = _connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_failure_incidents(machine_id: str, exclude_incident_id: Optional[int] = None,
                           db_path: Path = DB_PATH, as_of: Optional[str] = None) -> list[dict]:
    """as_of keeps only incidents STRICTLY BEFORE that moment (the past of this machine)."""
    sql = ("SELECT incident_id, date, failure_type, description, resolution, root_cause "
           "FROM failure_incidents WHERE machine_id = ?")
    params: list = [machine_id]
    if exclude_incident_id is not None:
        sql += " AND incident_id != ?"
        params.append(exclude_incident_id)
    if as_of is not None:
        sql += " AND date < ?"
        params.append(as_of)
    sql += " ORDER BY date DESC"
    conn = _connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_incident_with_reading(incident_id: int, db_path: Path = DB_PATH) -> Optional[dict]:
    """Join an incident back to its originating sensor reading (same technique as Phase 2)."""
    conn = _connect(db_path)
    row = conn.execute(
        """SELECT fi.incident_id, fi.machine_id, fi.date, fi.failure_type, fi.description,
                  sr.air_temperature, sr.process_temperature, sr.rotational_speed, sr.torque, sr.tool_wear
           FROM failure_incidents fi
           JOIN sensor_readings sr ON fi.machine_id = sr.machine_id AND fi.date = sr.timestamp
           WHERE fi.incident_id = ?""",
        (incident_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_incidents_with_readings(exclude_incident_id: Optional[int] = None,
                                     db_path: Path = DB_PATH) -> list[dict]:
    """Every failure incident joined to its sensor values — used for the sensor-distance ranking."""
    conn = _connect(db_path)
    query = """SELECT fi.incident_id, fi.machine_id, fi.date, fi.failure_type, fi.description,
                      sr.air_temperature, sr.process_temperature, sr.rotational_speed, sr.torque, sr.tool_wear
               FROM failure_incidents fi
               JOIN sensor_readings sr ON fi.machine_id = sr.machine_id AND fi.date = sr.timestamp"""
    params = ()
    if exclude_incident_id is not None:
        query += " WHERE fi.incident_id != ?"
        params = (exclude_incident_id,)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_channel_normalization_stats(db_path: Path = DB_PATH) -> dict:
    """Global mean/std per sensor channel, for z-score normalization in distance calcs."""
    conn = _connect(db_path)
    row = conn.execute(
        """SELECT AVG(air_temperature) am, AVG(process_temperature) pm, AVG(rotational_speed) rm,
                  AVG(torque) tm, AVG(tool_wear) wm
           FROM sensor_readings"""
    ).fetchone()

    import math
    channels = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    means = {"air_temperature": row["am"], "process_temperature": row["pm"],
             "rotational_speed": row["rm"], "torque": row["tm"], "tool_wear": row["wm"]}

    # Compute std in Python — older sqlite3 builds don't all have STDEV built in
    stds = {}
    for ch in channels:
        vals = [r[ch] for r in conn.execute(f"SELECT {ch} FROM sensor_readings").fetchall()]
        m = means[ch]
        variance = sum((v - m) ** 2 for v in vals) / len(vals)
        stds[ch] = math.sqrt(variance) or 1.0  # avoid divide-by-zero

    conn.close()
    return {"means": means, "stds": stds}
