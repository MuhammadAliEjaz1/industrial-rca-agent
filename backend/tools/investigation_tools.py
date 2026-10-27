"""
Phase 4: The 7 investigation tools.

Each tool is a thin wrapper: it gathers data from one source (DB, ML model, or
RAG), validates it into the matching Pydantic schema, and returns model_dump().
No tool touches more than one source — that keeps each one independently
testable and keeps the LangGraph agent (Phase 5) in charge of combining them.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool

from database import queries as db
from models.failure_classifier import predict_failure
from rag.retriever import (
    search_technical_documents as _rag_search_documents,
    search_similar_incidents as _rag_search_incidents,
)
from schemas.tool_schemas import (
    SensorDataResult, MaintenanceHistoryResult, FailureHistoryResult,
    AnomalyDetectionResult, SensorStatisticsResult,
    DocumentSearchResult, IncidentSearchResult,
)


# ─────────────────────────────────────────────
# 1. get_sensor_data
# ─────────────────────────────────────────────
@tool
def get_sensor_data(machine_id: str, limit: int = 20, as_of: Optional[str] = None) -> dict:
    """Get the most recent sensor readings for a machine (air/process temperature,
    rotational speed, torque, tool wear), newest first. as_of (optional timestamp text) hides
    later readings, for investigating a historical incident."""
    if not db.machine_exists(machine_id):
        return SensorDataResult(machine_id=machine_id, found=False, reading_count=0,
                                 error=f"No machine with id '{machine_id}'").model_dump()

    readings = db.get_recent_readings(machine_id, limit=limit, as_of=as_of)
    return SensorDataResult(
        machine_id=machine_id, found=True, reading_count=len(readings), readings=readings
    ).model_dump()


# ─────────────────────────────────────────────
# 2. get_maintenance_history
# ─────────────────────────────────────────────
@tool
def get_maintenance_history(machine_id: str, as_of: Optional[str] = None) -> dict:
    """Get maintenance records for a machine (scheduled and reactive), newest first.
    as_of (optional) keeps only records dated on or before that moment."""
    if not db.machine_exists(machine_id):
        return MaintenanceHistoryResult(machine_id=machine_id, found=False, record_count=0,
                                         error=f"No machine with id '{machine_id}'").model_dump()

    records = db.get_maintenance_records(machine_id, as_of=as_of)
    return MaintenanceHistoryResult(
        machine_id=machine_id, found=True, record_count=len(records), records=records
    ).model_dump()


# ─────────────────────────────────────────────
# 3. get_failure_history
# ─────────────────────────────────────────────
@tool
def get_failure_history(machine_id: str, exclude_incident_id: Optional[int] = None,
                        as_of: Optional[str] = None) -> dict:
    """Get past failure incidents for a machine, newest first. Pass exclude_incident_id
    when investigating that exact incident, so the agent can't see its own answer.
    as_of (optional) keeps only incidents strictly before that moment."""
    if not db.machine_exists(machine_id):
        return FailureHistoryResult(machine_id=machine_id, found=False, incident_count=0,
                                     error=f"No machine with id '{machine_id}'").model_dump()

    incidents = db.get_failure_incidents(machine_id, exclude_incident_id=exclude_incident_id, as_of=as_of)
    return FailureHistoryResult(
        machine_id=machine_id, found=True, incident_count=len(incidents), incidents=incidents
    ).model_dump()


# ─────────────────────────────────────────────
# 4. run_anomaly_detection
# ─────────────────────────────────────────────
@tool
def run_anomaly_detection(machine_id: str, as_of: Optional[str] = None) -> dict:
    """Run the trained XGBoost classifier on a machine's latest sensor reading (as of the given
    timestamp, if any). Returns
    the predicted failure class, confidence, full probability distribution, feature
    importances, and an honesty flag/note for classes the model is known to be weak on
    (see Phase 2 evaluation: Tool Wear Failure, Unknown, rare multi-cause combinations)."""
    if not db.machine_exists(machine_id):
        return AnomalyDetectionResult(machine_id=machine_id, found=False,
                                       error=f"No machine with id '{machine_id}'").model_dump()

    readings = db.get_recent_readings(machine_id, limit=1, as_of=as_of)
    if not readings:
        return AnomalyDetectionResult(machine_id=machine_id, found=False,
                                       error="Machine exists but has no sensor readings").model_dump()

    latest = readings[0]
    try:
        prediction = predict_failure(
            air_temperature=latest["air_temperature"],
            process_temperature=latest["process_temperature"],
            rotational_speed=latest["rotational_speed"],
            torque=latest["torque"],
            tool_wear=latest["tool_wear"],
        )
    except FileNotFoundError as e:
        return AnomalyDetectionResult(machine_id=machine_id, found=False,
                                       error=f"Model not trained yet: {e}").model_dump()

    # Surface a near-tie between top-2 classes (the gap the boolean flag alone misses —
    # see the 50.6% vs 48.0% Tool Wear Failure test in Phase 2).
    sorted_probs = sorted(prediction["class_probabilities"].items(), key=lambda x: -x[1])
    close_second_class, close_second_margin = None, None
    if len(sorted_probs) >= 2:
        close_second_class = sorted_probs[1][0]
        close_second_margin = round(sorted_probs[0][1] - sorted_probs[1][1], 4)

    return AnomalyDetectionResult(
        machine_id=machine_id, found=True,
        based_on_reading_timestamp=latest["timestamp"],
        close_second_class=close_second_class, close_second_margin=close_second_margin,
        **prediction,
    ).model_dump()


# ─────────────────────────────────────────────
# 5. search_technical_documents
# ─────────────────────────────────────────────
@tool
def search_technical_documents(query: str, k: int = 4, failure_type: Optional[str] = None) -> dict:
    """Semantic search over the technical manuals (failure-mode guides, machine spec,
    maintenance schedule). Optionally restrict to one failure_type's manual."""
    results = _rag_search_documents(query, k=k, failure_type=failure_type)
    return DocumentSearchResult(query=query, result_count=len(results), results=results).model_dump()


# ─────────────────────────────────────────────
# 6. calculate_sensor_statistics
# ─────────────────────────────────────────────
@tool
def calculate_sensor_statistics(machine_id: str, as_of: Optional[str] = None) -> dict:
    """Compute mean/std/min/max/latest and a simple trend (increasing/decreasing/stable)
    for each sensor channel, across a machine's full reading history (up to as_of, if given)."""
    if not db.machine_exists(machine_id):
        return SensorStatisticsResult(machine_id=machine_id, found=False, reading_count=0,
                                       error=f"No machine with id '{machine_id}'").model_dump()

    readings = db.get_all_readings(machine_id, as_of=as_of)  # chronological order
    if not readings:
        return SensorStatisticsResult(machine_id=machine_id, found=False, reading_count=0,
                                       error="Machine exists but has no sensor readings").model_dump()

    channels = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    channel_stats = {}
    for ch in channels:
        values = [r[ch] for r in readings]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance ** 0.5

        # Trend: compare the mean of the last 20% of readings vs the mean of the rest.
        # Threshold is based on the standard error of that two-segment difference
        # (multiplier=2.0, roughly a 95%-one-tailed cutoff) — NOT an arbitrary fraction
        # of std. An earlier version used 0.1*std and flagged pure random noise as a
        # trend on rotational_speed in testing (diff -19.5 vs threshold 2.6); this
        # version correctly reads noisy-but-flat channels as "stable".
        split = max(1, int(n * 0.8))
        early_vals, recent_vals = values[:split], values[split:] or values[-1:]
        early_mean = sum(early_vals) / len(early_vals)
        recent_mean = sum(recent_vals) / len(recent_vals)
        se = std * ((1 / len(early_vals) + 1 / len(recent_vals)) ** 0.5) if std > 0 else 0
        threshold = 2.0 * se
        diff = recent_mean - early_mean
        if diff > threshold:
            trend = "increasing"
        elif -diff > threshold:
            trend = "decreasing"
        else:
            trend = "stable"

        channel_stats[ch] = {
            "mean": round(mean, 3), "std": round(std, 3),
            "min": round(min(values), 3), "max": round(max(values), 3),
            "latest": round(values[-1], 3), "trend": trend,
        }

    return SensorStatisticsResult(
        machine_id=machine_id, found=True, reading_count=len(readings), channels=channel_stats
    ).model_dump()


# ─────────────────────────────────────────────
# 7. search_similar_incidents  (hybrid: SQL numeric distance + semantic text)
# ─────────────────────────────────────────────
def _sensor_distance(a: dict, b: dict, means: dict, stds: dict, channels: list[str]) -> float:
    """Euclidean distance in z-score space, so no single channel (e.g. rotational_speed,
    which ranges in the thousands) dominates just because its raw units are bigger."""
    total = 0.0
    for ch in channels:
        za = (a[ch] - means[ch]) / stds[ch]
        zb = (b[ch] - means[ch]) / stds[ch]
        total += (za - zb) ** 2
    return total ** 0.5


@tool
def search_similar_incidents(
    description: str,
    machine_id: Optional[str] = None,
    exclude_incident_id: Optional[int] = None,
    air_temperature: Optional[float] = None,
    process_temperature: Optional[float] = None,
    rotational_speed: Optional[float] = None,
    torque: Optional[float] = None,
    tool_wear: Optional[float] = None,
    k: int = 5,
) -> dict:
    """Find historically similar failure incidents. If the 5 sensor values are given,
    ranks by sensor-profile distance (the reliable signal — see Phase 3 finding that
    embeddings barely capture numeric similarity). Otherwise falls back to semantic
    text search over incident descriptions."""
    channels = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    query_values = {
        "air_temperature": air_temperature, "process_temperature": process_temperature,
        "rotational_speed": rotational_speed, "torque": torque, "tool_wear": tool_wear,
    }
    has_full_sensor_profile = all(v is not None for v in query_values.values())

    if has_full_sensor_profile:
        norm = db.get_channel_normalization_stats()
        candidates = db.get_all_incidents_with_readings(exclude_incident_id=exclude_incident_id)
        if machine_id:
            candidates = [c for c in candidates if c["machine_id"] == machine_id]

        for c in candidates:
            c["sensor_distance"] = round(
                _sensor_distance(query_values, c, norm["means"], norm["stds"], channels), 4
            )
        candidates.sort(key=lambda c: c["sensor_distance"])
        top = candidates[:k]

        results = [{
            "incident_id": c["incident_id"], "machine_id": c["machine_id"], "date": c["date"],
            "failure_type": c["failure_type"], "content": c["description"],
            "semantic_similarity": None,  # not computed on this path (sensor-distance ranking used instead)
            "sensor_distance": c["sensor_distance"],
        } for c in top]

        return IncidentSearchResult(
            description=description, result_count=len(results),
            ranked_by="sensor_distance", results=results,
        ).model_dump()

    # Fallback: no sensor profile given — semantic search only (weaker, per Phase 3 finding)
    semantic_results = _rag_search_incidents(description, k=k, machine_id=machine_id,
                                              exclude_incident_id=exclude_incident_id)
    results = [{
        "incident_id": r["incident_id"], "machine_id": r["machine_id"], "date": r["date"],
        "failure_type": r["failure_type"], "content": r["content"],
        "semantic_similarity": r["similarity"], "sensor_distance": None,
    } for r in semantic_results]

    return IncidentSearchResult(
        description=description, result_count=len(results),
        ranked_by="semantic_similarity", results=results,
    ).model_dump()


ALL_TOOLS = [
    get_sensor_data, get_maintenance_history, get_failure_history,
    run_anomaly_detection, search_technical_documents,
    calculate_sensor_statistics, search_similar_incidents,
]
