"""Read-only database access for the API. The API never writes to industrial_rca.db.

It reuses the already-tested SQL helpers in database/queries.py where they exist, and adds only the few
aggregate queries the dashboard needs. The 'machines' table's exact columns are not assumed: every column other
than machine_id is passed through as an attribute."""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from database import queries as q  # noqa: E402  (frozen module, only called, never changed)
from graph.checks import compute_limit_checks  # noqa: E402

DEFAULT_DB = q.DB_PATH
SENSORS = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]


def _connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")   # belt and braces: this connection cannot write
    return conn


def limit_status(reading: Optional[dict]) -> Optional[dict]:
    """Deterministic rule check of one reading, in the shape of the LimitStatus model."""
    if not reading:
        return None
    checks = compute_limit_checks([{k: float(reading[k]) for k in SENSORS}])
    latest = checks["latest"]
    return {
        "temp_diff_k": latest["temp_diff_k"], "power_w": latest["power_w"], "wear_x_torque": latest["wear_x_torque"],
        "heat_dissipation": latest["heat_dissipation"], "power_out_of_range": latest["power_out_of_range"],
        "possible_overstrain": latest["possible_overstrain"],
        "tool_wear_at_failure_range": latest["tool_wear_at_failure_range"],
        "hard_limit_violated": bool(checks["violated"]),
    }


def status_of(status: Optional[dict]) -> str:
    if status is None:
        return "no_data"
    if status["hard_limit_violated"]:
        return "limit_violated"
    return "tool_wear_hint" if status["tool_wear_at_failure_range"] else "ok"


def _summary_from_row(row: sqlite3.Row, latest: Optional[dict]) -> dict:
    d = dict(row)
    attrs = {k: v for k, v in d.items() if not k.startswith("__") and k != "machine_id"}
    st = limit_status(latest)
    return {
        "machine_id": d["machine_id"], "attributes": attrs, "n_readings": d["__n_readings"],
        "latest_reading_at": d["__latest_reading_at"], "n_incidents": d["__n_incidents"],
        "n_maintenance": d["__n_maintenance"], "latest_status": status_of(st),
    }


_SUMMARY_SQL = """
    SELECT m.*,
      (SELECT COUNT(*) FROM sensor_readings s WHERE s.machine_id = m.machine_id)       AS __n_readings,
      (SELECT MAX(timestamp) FROM sensor_readings s WHERE s.machine_id = m.machine_id) AS __latest_reading_at,
      (SELECT COUNT(*) FROM failure_incidents f WHERE f.machine_id = m.machine_id)     AS __n_incidents,
      (SELECT COUNT(*) FROM maintenance_records r WHERE r.machine_id = m.machine_id)   AS __n_maintenance
    FROM machines m
"""


def list_machines(db_path=DEFAULT_DB) -> list[dict]:
    conn = _connect(db_path)
    rows = conn.execute(_SUMMARY_SQL + " ORDER BY m.machine_id").fetchall()
    latest = {r["machine_id"]: dict(r) for r in conn.execute(
        "SELECT s.* FROM sensor_readings s JOIN (SELECT machine_id, MAX(timestamp) AS ts FROM sensor_readings "
        "GROUP BY machine_id) l ON s.machine_id = l.machine_id AND s.timestamp = l.ts")}
    conn.close()
    return [_summary_from_row(r, latest.get(r["machine_id"])) for r in rows]


def get_machine(machine_id: str, db_path=DEFAULT_DB) -> Optional[dict]:
    conn = _connect(db_path)
    row = conn.execute(_SUMMARY_SQL + " WHERE m.machine_id = ?", (machine_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    recent = q.get_recent_readings(machine_id, limit=1, db_path=db_path)
    latest = recent[0] if recent else None
    out = _summary_from_row(row, latest)
    out["latest_reading"] = latest
    out["limit_status"] = limit_status(latest)
    return out


def machine_exists(machine_id: str, db_path=DEFAULT_DB) -> bool:
    return q.machine_exists(machine_id, db_path=db_path)


def sensor_series(machine_id: str, limit: int, as_of: Optional[str], db_path=DEFAULT_DB) -> list[dict]:
    rows = q.get_recent_readings(machine_id, limit=limit, db_path=db_path, as_of=as_of)   # newest first
    return list(reversed(rows))                                                           # chart order: oldest first


def maintenance(machine_id: str, as_of: Optional[str], db_path=DEFAULT_DB) -> list[dict]:
    return q.get_maintenance_records(machine_id, db_path=db_path, as_of=as_of)


def incidents(machine_id: str, as_of: Optional[str], db_path=DEFAULT_DB) -> list[dict]:
    return q.get_failure_incidents(machine_id, db_path=db_path, as_of=as_of)
