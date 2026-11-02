"""Request and response models for the API (Pydantic v2). These are what the React dashboard will consume."""

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DATA_NOTICE = [
    "Machine IDs, timestamps and maintenance records are SYNTHETIC. Sensor values and failure labels come from the "
    "AI4I 2020 dataset.",
    "The readings of one machine are independent samples, not a real time series: do not read trends into them.",
]

_TS = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?$")


# ---------- machines ----------

class SensorReading(BaseModel):
    timestamp: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


class LimitStatus(BaseModel):
    """Deterministic rule check of ONE reading (the documented limits in the manuals). No ML, no LLM."""
    temp_diff_k: float
    power_w: float
    wear_x_torque: float
    heat_dissipation: bool
    power_out_of_range: bool
    possible_overstrain: bool
    tool_wear_at_failure_range: bool
    hard_limit_violated: bool


MachineStatus = Literal["ok", "limit_violated", "tool_wear_hint", "no_data"]


class MachineSummary(BaseModel):
    machine_id: str
    attributes: dict[str, Any] = Field(default_factory=dict, description="Any other columns of the machines table.")
    n_readings: int
    latest_reading_at: Optional[str] = None
    n_incidents: int
    n_maintenance: int
    latest_status: MachineStatus = Field(
        description="Rule check of the latest reading only: limit_violated = a hard limit is met, "
                    "tool_wear_hint = tool wear >= 200 min (a weak signal), ok, or no_data.")


class MachineDetail(MachineSummary):
    latest_reading: Optional[SensorReading] = None
    limit_status: Optional[LimitStatus] = None


class SensorSeries(BaseModel):
    machine_id: str
    count: int
    readings: list[SensorReading] = Field(description="Oldest first, ready for a chart.")
    notices: list[str] = DATA_NOTICE


class MaintenanceRecord(BaseModel):
    date: str
    maintenance_type: Optional[str] = None
    description: Optional[str] = None
    technician: Optional[str] = None
    parts_replaced: Optional[str] = None


class MaintenanceHistory(BaseModel):
    machine_id: str
    count: int
    records: list[MaintenanceRecord]
    notices: list[str] = DATA_NOTICE


class IncidentRecord(BaseModel):
    incident_id: int
    date: str
    failure_type: Optional[str] = None
    description: Optional[str] = None
    resolution: Optional[str] = None
    root_cause: Optional[str] = None


class IncidentHistory(BaseModel):
    machine_id: str
    count: int
    incidents: list[IncidentRecord]


# ---------- investigations ----------

class InvestigationRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=40)
    failure_reported: bool = Field(
        default=True, description="True: a failure/stoppage was reported. False: routine check (health-check mode).")
    description: Optional[str] = Field(default=None, max_length=500, description="Free text shown to the agent.")
    as_of: Optional[str] = Field(
        default=None, description="Investigate as if 'now' were this timestamp (data after it is hidden). "
                                  "Format 'YYYY-MM-DD HH:MM:SS[.ffffff]'. Used for demos and evaluation.")
    incident_id: Optional[int] = Field(
        default=None, description="Hide this historical incident from the agent's own evidence search "
                                  "(for demos and evaluation).")

    @field_validator("as_of")
    @classmethod
    def _check_as_of(cls, v):
        if v is None:
            return v
        if not _TS.match(v):
            raise ValueError("as_of must look like 'YYYY-MM-DD HH:MM:SS' (optionally with .ffffff)")
        return v.replace("T", " ")


class LogEntry(BaseModel):
    step: str
    status: str
    summary: str
    timestamp: str


class CandidateCause(BaseModel):
    model_config = ConfigDict(extra="allow")
    cause: str
    confidence: str
    supporting_evidence: list[str] = []
    contradicting_evidence: list[str] = []
    reasoning: Optional[str] = None
    llm_confidence: Optional[str] = Field(default=None, description="The model's own rating, kept as a second opinion.")
    confidence_basis: Optional[str] = Field(default=None, description="Why the documented rules set this confidence.")
    evaluated_by: Optional[str] = Field(default=None, description="'rule' if it was settled without asking the model.")


class InvestigationReport(BaseModel):
    """The final report. Unlisted keys (limit_checks, ml_analysis, machine_condition, ...) pass through unchanged."""
    model_config = ConfigDict(extra="allow")
    mode: Optional[str] = None
    most_supported_cause: Optional[str] = None
    confidence_level: Optional[str] = None
    justification: Optional[str] = None
    observed_facts: list[str] = []
    ml_predictions: list[str] = []
    advisories: list[str] = []
    unsupported_speculation: list[str] = Field(default=[], description="Hypotheses that were evaluated and not chosen.")
    candidate_root_causes: list[CandidateCause] = []
    evidence_sources: list[str] = []


RunStatus = Literal["queued", "running", "done", "error"]


class InvestigationAccepted(BaseModel):
    investigation_id: str
    status: RunStatus
    machine_id: str
    status_url: str
    events_url: str


class InvestigationStatus(BaseModel):
    investigation_id: str
    status: RunStatus
    machine_id: str
    request: InvestigationRequest
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log: list[LogEntry] = []
    report: Optional[InvestigationReport] = None
    error: Optional[str] = None
    error_kind: Optional[str] = Field(default=None, description="e.g. 'daily_token_limit', 'budget', 'internal'.")
    notices: list[str] = DATA_NOTICE


class InvestigationSummary(BaseModel):
    investigation_id: str
    status: RunStatus
    machine_id: str
    created_at: str
    finished_at: Optional[str] = None
    most_supported_cause: Optional[str] = None
    confidence_level: Optional[str] = None
