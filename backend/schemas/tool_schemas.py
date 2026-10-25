"""
Pydantic response models for every investigation tool.
Structured output, per the project's requirement to avoid uncontrolled strings.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    timestamp: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


class SensorDataResult(BaseModel):
    machine_id: str
    found: bool
    reading_count: int
    readings: list[SensorReading] = Field(default_factory=list)
    error: Optional[str] = None


class MaintenanceRecord(BaseModel):
    date: str
    maintenance_type: str
    description: str
    technician: str
    parts_replaced: Optional[str] = None


class MaintenanceHistoryResult(BaseModel):
    machine_id: str
    found: bool
    record_count: int
    records: list[MaintenanceRecord] = Field(default_factory=list)
    error: Optional[str] = None


class FailureIncidentRecord(BaseModel):
    incident_id: int
    date: str
    failure_type: str
    description: str
    resolution: Optional[str] = None
    root_cause: Optional[str] = None


class FailureHistoryResult(BaseModel):
    machine_id: str
    found: bool
    incident_count: int
    incidents: list[FailureIncidentRecord] = Field(default_factory=list)
    error: Optional[str] = None


class AnomalyDetectionResult(BaseModel):
    machine_id: str
    found: bool
    based_on_reading_timestamp: Optional[str] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    anomaly_score: Optional[float] = None
    class_probabilities: Optional[dict[str, float]] = None
    top_features: Optional[list[dict]] = None
    reliability_warning: Optional[bool] = None
    reliability_note: Optional[str] = None
    close_second_class: Optional[str] = None
    close_second_margin: Optional[float] = Field(
        default=None,
        description="Probability gap between the top and second class. Under ~0.10 means "
                    "the model is effectively torn between two hypotheses (see Phase 2 test where "
                    "'No Failure' beat 'Tool Wear Failure' by only 2.6 points)."
    )
    error: Optional[str] = None


class ChannelStatistics(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    latest: float
    trend: str  # "increasing" | "decreasing" | "stable"


class SensorStatisticsResult(BaseModel):
    machine_id: str
    found: bool
    reading_count: int
    channels: dict[str, ChannelStatistics] = Field(default_factory=dict)
    error: Optional[str] = None


class DocumentSearchResultItem(BaseModel):
    content: str
    source: str
    title: str
    section: str
    failure_type: str
    similarity: float


class DocumentSearchResult(BaseModel):
    query: str
    result_count: int
    results: list[DocumentSearchResultItem] = Field(default_factory=list)


class IncidentSearchResultItem(BaseModel):
    incident_id: int
    machine_id: str
    date: str
    failure_type: str
    content: str
    semantic_similarity: Optional[float] = None
    sensor_distance: Optional[float] = Field(
        default=None,
        description="Normalized Euclidean distance across the 5 sensor channels vs. the query "
                    "reading. Lower = more similar sensor profile. None if no sensor values were given."
    )


class IncidentSearchResult(BaseModel):
    description: str
    result_count: int
    ranked_by: str  # "sensor_distance" | "semantic_similarity"
    results: list[IncidentSearchResultItem] = Field(default_factory=list)
