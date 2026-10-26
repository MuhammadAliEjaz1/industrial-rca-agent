"""
Phase 5: Graph state and the Pydantic schemas used for structured LLM output.

Reducer note: only investigation_log accumulates (Annotated + add) — it's the
timeline the frontend will show. Every other field is a plain overwrite, set
once by the node responsible for it. This matches the course's own rule:
give a field a reducer only when it should accumulate, not by default.
"""

from typing import Annotated, Literal, Optional, TypedDict
from operator import add
from pydantic import BaseModel, Field


class InvestigationState(TypedDict):
    # Input
    machine_id: str
    incident_description: str
    incident_id: Optional[int]  # set when investigating a KNOWN historical incident (Phase 8 eval);
                                 # None for a fresh, live investigation
    failure_reported: bool       # True = a failure/stoppage was reported (default if omitted);
                                 # False = routine check, nothing reported -> health-check mode

    # Gathered evidence (filled in by gather_context / run_anomaly_model / gather_evidence)
    sensor_readings: list[dict]
    sensor_statistics: dict
    anomaly_result: dict
    maintenance_records: list[dict]
    historical_incidents: list[dict]
    similar_incidents: list[dict]

    # Hypothesis work
    hypotheses: list[dict]               # from generate_hypotheses
    hypothesis_evaluations: list[dict]   # from investigate_hypotheses (evidence for/against each)
    retry_count: int

    # Conclusion
    root_cause: dict
    final_report: dict

    # Frontend timeline
    investigation_log: Annotated[list[dict], add]


# ─────────────────────────────────────────────
# Structured LLM outputs (what .with_structured_output() parses into)
# ─────────────────────────────────────────────

class HypothesisItem(BaseModel):
    cause: str = Field(description="EXACTLY one label from the allowed list given in the prompt. "
                                    "No other wording, no explanation here.")
    rationale: str = Field(description="Why this is plausible given the computed limit checks and the "
                                        "ML prediction. All explanation goes here.")


class HypothesisList(BaseModel):
    hypotheses: list[HypothesisItem] = Field(
        description="1 to 4 distinct candidate causes, each an allowed label from the prompt. Do NOT "
                    "include a 'no fault' or 'unexplained' entry; the system adds that itself."
    )


class HypothesisEvaluation(BaseModel):
    cause: str
    supporting_evidence: list[str] = Field(
        description="Specific facts FROM THE EVIDENCE PROVIDED that support this cause. "
                    "Empty list if none — do not invent evidence."
    )
    contradicting_evidence: list[str] = Field(
        description="Specific facts FROM THE EVIDENCE PROVIDED that contradict this cause. "
                    "Empty list if none."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="'high' only if multiple pieces of evidence clearly and specifically support this "
                    "cause with no major contradictions. 'low' if evidence is thin, generic, or mixed."
    )
    reasoning: str


class RootCauseDetermination(BaseModel):
    most_supported_cause: str = Field(
        description="The best-supported hypothesis's cause, verbatim. If no hypothesis clearly stands "
                    "out, use 'Unresolved - insufficient evidence' rather than guessing."
    )
    confidence_level: Literal["low", "medium", "high"]
    justification: str
    observed_facts: list[str] = Field(
        description="Plain facts read directly from sensor/maintenance/incident data. No interpretation."
    )
    ml_predictions: list[str] = Field(
        description="What the ML classifier predicted, stated explicitly as a PREDICTION, not a fact. "
                    "Include its reliability_warning/note if the predicted class is a known-weak one."
    )
    unsupported_speculation: list[str] = Field(
        description="Plausible-sounding ideas that the evidence did NOT support — list them as ruled "
                    "out or unconfirmed, don't just silently drop them."
    )
