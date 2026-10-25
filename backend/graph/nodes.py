"""
Phase 5: Graph nodes. Each node calls one or more Phase 4 tools directly as plain
Python functions (NOT via LLM tool-calling) — this graph is the fixed investigation
pipeline from Phase 0's design, not a dynamic ReAct loop where the LLM freely picks
tools. The LLM is used for the three genuinely judgment-requiring steps: hypothesis
generation, per-hypothesis evidence evaluation, and final root-cause determination.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_groq import ChatGroq

from tools.investigation_tools import (
    get_sensor_data, get_maintenance_history, get_failure_history,
    run_anomaly_detection, search_technical_documents,
    calculate_sensor_statistics, search_similar_incidents,
)
from graph.state import HypothesisList, HypothesisEvaluation, RootCauseDetermination
from dotenv import load_dotenv
load_dotenv()

MAX_RETRIES = 2

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="openai/gpt-oss-120b")
    return _llm


def _log(step: str, status: str, summary: str) -> dict:
    return {"investigation_log": [{
        "step": step, "status": status, "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]}


# ─────────────────────────────────────────────
# 1. gather_context
# ─────────────────────────────────────────────
def gather_context(state: dict) -> dict:
    machine_id = state["machine_id"]
    sensor_data = get_sensor_data.invoke({"machine_id": machine_id, "limit": 20})
    stats = calculate_sensor_statistics.invoke({"machine_id": machine_id})

    if not sensor_data["found"]:
        return {
            "sensor_readings": [], "sensor_statistics": stats,
            **_log("gather_context", "error", sensor_data.get("error", "unknown error")),
        }

    return {
        "sensor_readings": sensor_data["readings"],
        "sensor_statistics": stats,
        **_log("gather_context", "done",
               f"Loaded {sensor_data['reading_count']} recent readings for {machine_id}."),
    }


# ─────────────────────────────────────────────
# 2. run_anomaly_model
# ─────────────────────────────────────────────
def run_anomaly_model(state: dict) -> dict:
    result = run_anomaly_detection.invoke({"machine_id": state["machine_id"]})
    summary = (f"Predicted: {result.get('predicted_class')} "
               f"(confidence {result.get('confidence', 0):.2f})" if result.get("found")
               else f"Could not run model: {result.get('error')}")
    return {"anomaly_result": result, **_log("run_anomaly_model", "done", summary)}


# ─────────────────────────────────────────────
# 3. gather_evidence
# ─────────────────────────────────────────────
def gather_evidence(state: dict) -> dict:
    machine_id = state["machine_id"]
    exclude_id = state.get("incident_id")

    maintenance = get_maintenance_history.invoke({"machine_id": machine_id})
    history = get_failure_history.invoke({"machine_id": machine_id, "exclude_incident_id": exclude_id})

    similar = {"results": []}
    if state.get("sensor_readings"):
        latest = state["sensor_readings"][0]
        similar = search_similar_incidents.invoke({
            "description": state.get("incident_description", ""),
            "exclude_incident_id": exclude_id,
            "air_temperature": latest["air_temperature"],
            "process_temperature": latest["process_temperature"],
            "rotational_speed": latest["rotational_speed"],
            "torque": latest["torque"],
            "tool_wear": latest["tool_wear"],
            "k": 5,
        })

    return {
        "maintenance_records": maintenance.get("records", []),
        "historical_incidents": history.get("incidents", []),
        "similar_incidents": similar.get("results", []),
        **_log("gather_evidence", "done",
               f"{maintenance.get('record_count', 0)} maintenance records, "
               f"{history.get('incident_count', 0)} past incidents, "
               f"{len(similar.get('results', []))} similar incidents."),
    }


# ─────────────────────────────────────────────
# 4. generate_hypotheses
# ─────────────────────────────────────────────
def generate_hypotheses(state: dict) -> dict:
    anomaly = state.get("anomaly_result", {})
    stats = state.get("sensor_statistics", {}).get("channels", {})
    trends = [f"{ch}: {s['trend']} (latest={s['latest']})" for ch, s in stats.items()]

    prompt = f"""You are investigating a possible industrial machine failure. Propose candidate root causes.

Incident description: {state.get('incident_description', '(none provided)')}

ML classifier prediction: {anomaly.get('predicted_class')} (confidence {anomaly.get('confidence')})
{"WARNING: " + anomaly.get('reliability_note') if anomaly.get('reliability_warning') else ""}
Close second prediction: {anomaly.get('close_second_class')} (margin {anomaly.get('close_second_margin')})

Sensor trends:
{chr(10).join(trends) if trends else "No trend data available."}

Recent maintenance: {len(state.get('maintenance_records', []))} records on file.
Past failures on this machine: {len(state.get('historical_incidents', []))} incidents on file.

Propose 2-4 distinct candidate causes. The ML prediction is a STARTING hypothesis, not a confirmed
answer — if it carries a reliability warning, treat it with appropriate skepticism and still consider
alternative explanations. Include an "unexplained" hypothesis if nothing clearly fits."""

    llm = get_llm().with_structured_output(HypothesisList)
    result: HypothesisList = llm.invoke(prompt)
    hypotheses = [h.model_dump() for h in result.hypotheses]

    return {
        "hypotheses": hypotheses,
        **_log("generate_hypotheses", "done",
               f"Generated {len(hypotheses)} hypotheses: " + ", ".join(h["cause"] for h in hypotheses)),
    }


# ─────────────────────────────────────────────
# 5. investigate_hypotheses
# ─────────────────────────────────────────────
def investigate_hypotheses(state: dict) -> dict:
    maintenance_text = "\n".join(
        f"- {r['date']}: {r['maintenance_type']} - {r['description']}"
        for r in state.get("maintenance_records", [])
    ) or "(none)"

    similar_text = "\n".join(
        f"- Incident {r['incident_id']} ({r['machine_id']}, {r['date']}): {r['failure_type']} "
        f"[sensor distance: {r.get('sensor_distance')}]"
        for r in state.get("similar_incidents", [])
    ) or "(none)"

    stats = state.get("sensor_statistics", {}).get("channels", {})
    stats_text = "\n".join(f"- {ch}: {s}" for ch, s in stats.items()) or "(none)"

    evaluations = []
    for hyp in state.get("hypotheses", []):
        docs = search_technical_documents.invoke({"query": hyp["cause"], "k": 3})
        docs_text = "\n".join(
            f"- [{d['source']} > {d['section']}] {d['content'][:400]}"
            for d in docs.get("results", [])
        ) or "(no relevant documentation found)"

        prompt = f"""Evaluate this candidate root cause against the evidence below. Be skeptical —
only cite something as supporting or contradicting evidence if it specifically relates to this cause.

CANDIDATE CAUSE: {hyp['cause']}
Rationale given: {hyp['rationale']}

SENSOR STATISTICS:
{stats_text}

TECHNICAL DOCUMENTATION (retrieved for this specific cause):
{docs_text}

MAINTENANCE HISTORY:
{maintenance_text}

SIMILAR PAST INCIDENTS (ranked by sensor-profile similarity):
{similar_text}

Evaluate: what supports this cause, what contradicts it, and how confident should we be?"""

        llm = get_llm().with_structured_output(HypothesisEvaluation)
        result: HypothesisEvaluation = llm.invoke(prompt)
        evaluations.append(result.model_dump())

    confidences = [e["confidence"] for e in evaluations]
    summary = ", ".join(f"{e['cause']} ({e['confidence']})" for e in evaluations)

    return {
        "hypothesis_evaluations": evaluations,
        **_log("investigate_hypotheses", "done", f"Evaluated {len(evaluations)} hypotheses: {summary}"),
    }


def route_after_investigation(state: dict) -> str:
    """Conditional edge: loop back for more evidence, or move on to a conclusion."""
    evaluations = state.get("hypothesis_evaluations", [])
    has_high_confidence = any(e["confidence"] == "high" for e in evaluations)
    retry_count = state.get("retry_count", 0)

    if has_high_confidence or retry_count >= MAX_RETRIES:
        return "sufficient"
    return "insufficient"


# ─────────────────────────────────────────────
# 6. gather_more_evidence (only runs on the "insufficient" branch)
# ─────────────────────────────────────────────
def gather_more_evidence(state: dict) -> dict:
    """Broaden the search: more documents per hypothesis, more similar incidents, and
    widen the incident search beyond this one machine (other machines of the same
    model can still be informative). Then loops back to investigate_hypotheses."""
    machine_id = state["machine_id"]
    retry_count = state.get("retry_count", 0) + 1

    broader_similar = {"results": []}
    if state.get("sensor_readings"):
        latest = state["sensor_readings"][0]
        broader_similar = search_similar_incidents.invoke({
            "description": state.get("incident_description", ""),
            "exclude_incident_id": state.get("incident_id"),
            "air_temperature": latest["air_temperature"],
            "process_temperature": latest["process_temperature"],
            "rotational_speed": latest["rotational_speed"],
            "torque": latest["torque"],
            "tool_wear": latest["tool_wear"],
            "k": 10,  # widened from 5
        })

    return {
        "similar_incidents": broader_similar.get("results", []),
        "retry_count": retry_count,
        **_log("gather_more_evidence", "done",
               f"Retry {retry_count}/{MAX_RETRIES}: widened similar-incident search to "
               f"{len(broader_similar.get('results', []))} results."),
    }


# ─────────────────────────────────────────────
# 7. determine_root_cause
# ─────────────────────────────────────────────
def determine_root_cause(state: dict) -> dict:
    evaluations = state.get("hypothesis_evaluations", [])
    evaluations_text = "\n\n".join(
        f"Cause: {e['cause']}\nConfidence: {e['confidence']}\n"
        f"Supporting: {e['supporting_evidence']}\nContradicting: {e['contradicting_evidence']}\n"
        f"Reasoning: {e['reasoning']}"
        for e in evaluations
    ) or "(no hypotheses were evaluated)"

    anomaly = state.get("anomaly_result", {})

    prompt = f"""Based on the investigated hypotheses below, determine the most likely root cause.

HYPOTHESIS EVALUATIONS:
{evaluations_text}

ML CLASSIFIER CONTEXT (a prediction, not a confirmed fact):
predicted_class={anomaly.get('predicted_class')}, confidence={anomaly.get('confidence')}
reliability_warning={anomaly.get('reliability_warning')}: {anomaly.get('reliability_note')}

Separate observed facts (directly from sensor/maintenance/incident records) from ML predictions from
unsupported speculation. If no hypothesis has strong, specific support, say so explicitly — do not
present a weakly-supported guess as a confident conclusion."""

    llm = get_llm().with_structured_output(RootCauseDetermination)
    result: RootCauseDetermination = llm.invoke(prompt)

    return {
        "root_cause": result.model_dump(),
        **_log("determine_root_cause", "done",
               f"Most supported cause: {result.most_supported_cause} ({result.confidence_level})"),
    }


# ─────────────────────────────────────────────
# 8. generate_report
# ─────────────────────────────────────────────
def generate_report(state: dict) -> dict:
    """Mostly programmatic assembly of already-structured data — no new LLM call needed,
    since determine_root_cause already produced the key narrative judgment. Assembling
    the report in code (not another LLM pass) guarantees it can't drift from the
    structured facts we already validated."""
    anomaly = state.get("anomaly_result", {})
    root_cause = state.get("root_cause", {})

    evidence_sources = []
    for r in state.get("similar_incidents", []):
        evidence_sources.append(f"Incident {r['incident_id']} ({r['machine_id']}, {r['date']})")
    seen_docs = set()
    for hyp_eval in state.get("hypothesis_evaluations", []):
        pass  # doc sources were already folded into supporting_evidence text by the LLM

    report = {
        "incident_summary": state.get("incident_description", "(no description provided)"),
        "machine_condition": {
            "machine_id": state["machine_id"],
            "latest_reading": state["sensor_readings"][0] if state.get("sensor_readings") else None,
            "sensor_statistics": state.get("sensor_statistics", {}).get("channels", {}),
        },
        "ml_analysis": {
            "predicted_class": anomaly.get("predicted_class"),
            "confidence": anomaly.get("confidence"),
            "anomaly_score": anomaly.get("anomaly_score"),
            "reliability_warning": anomaly.get("reliability_warning"),
            "reliability_note": anomaly.get("reliability_note"),
            "close_second_class": anomaly.get("close_second_class"),
            "close_second_margin": anomaly.get("close_second_margin"),
        },
        "maintenance_history": state.get("maintenance_records", []),
        "candidate_root_causes": state.get("hypothesis_evaluations", []),
        "most_supported_cause": root_cause.get("most_supported_cause"),
        "confidence_level": root_cause.get("confidence_level"),
        "justification": root_cause.get("justification"),
        "observed_facts": root_cause.get("observed_facts", []),
        "ml_predictions": root_cause.get("ml_predictions", []),
        "unsupported_speculation": root_cause.get("unsupported_speculation", []),
        "evidence_sources": evidence_sources,
        "investigation_log": state.get("investigation_log", []),
    }

    return {
        "final_report": report,
        **_log("generate_report", "done", "Report assembled."),
    }
