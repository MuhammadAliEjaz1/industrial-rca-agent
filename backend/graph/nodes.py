"""
Phase 5: Graph nodes. Each node calls Phase 4 tools directly as plain Python functions
(NOT via LLM tool-calling): this graph is a fixed investigation pipeline, not a ReAct loop.
The LLM is used for three judgment steps: hypothesis generation, per-hypothesis evidence
evaluation, and final root-cause determination.

REVISION 4 (after the first Phase 8 pilot: 4/10 incidents exact vs 9/10 for the rules+classifier baseline): the code
  now sets confidence wherever a documented rule exists, decides the final outcome, and the LLM explains it.
  Violations are judged on the LATEST reading, tool wear in a routine check is an advisory, retries are off.
REVISION 3: hypotheses come from a CLOSED catalog (graph/checks.py), the evidence can force extra ones in,
  trends are no longer shown to the LLM, and observed facts that only repeat computed numbers are dropped.
REVISION 2 (after the first real Groq runs on machine M001, ML = No Failure):
  - Rules from the manuals (temperature gap, derived power, wear x torque, tool-wear range) are
    computed in graph/checks.py and handed to the LLM as facts, instead of the LLM doing the maths.
  - 'high' confidence needs a documented limit violation (capped in code, logged in the timeline).
  - A healthy machine with nothing violated is a HEALTH CHECK, not a failure investigation.
  - Retries only run when some hypothesis is 'medium' (promising); all-low means more of the
    same evidence will not help.
  - The final answer can never be more confident than the best hypothesis; if no hypothesis
    reaches 'medium' the answer is 'Unresolved' (or 'No fault identified' for a health check).
  - ML predictions in the report are written by code, and numbers in 'observed facts' that appear
    nowhere in the evidence are tagged [UNVERIFIED NUMBER(S)].
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

from tools.investigation_tools import (
    get_sensor_data, get_maintenance_history, get_failure_history,
    run_anomaly_detection, search_technical_documents,
    calculate_sensor_statistics, search_similar_incidents,
)
from graph.state import HypothesisList, HypothesisEvaluation, RootCauseDetermination
from graph.checks import (
    ORDER, DOC_CONSTANTS_TEXT, FAULT_CATALOG, H_NO_FAULT, compute_limit_checks, limit_check_facts, near_tie,
    is_health_check, is_null_hypothesis, rule_confidence, ml_agrees, decide_confidence, decide_final, tool_wear_advisory, annotate_ungrounded,
    finalize_hypotheses,
    drop_redundant_facts,
)

# The retry loop only widens the similar-incident list, which cannot change what the rules decide, and the Phase 8
# pilot showed it just cost time (about 0.6 retries per case). It stays in the graph but is switched off.
MAX_RETRIES = 0

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        # temperature=0 narrows run-to-run variation (the default is 0.7); it does not make it deterministic.
        kwargs = {"model": "openai/gpt-oss-120b", "temperature": 0}
        # Reasoning tokens count against Groq's token limits (free plan: 8K/minute, 200K/day). 'low' cuts tokens and
        # time; set GROQ_REASONING_EFFORT=none in .env to switch it off, or medium/high to raise it.
        effort = os.getenv("GROQ_REASONING_EFFORT", "low").strip().lower()
        if effort not in ("", "none", "off"):
            kwargs["reasoning_effort"] = effort
        _llm = ChatGroq(**kwargs)
    return _llm


def _invoke_structured(schema, prompt: str, attempts: int = 3):
    """Structured output through Groq tool-calling sometimes fails with 'tool_use_failed': the model answers in
    prose or a markdown table instead of calling the function (seen in a real run). Retry with a stricter
    reminder; any other error is raised immediately."""
    llm = get_llm().with_structured_output(schema)
    last_error = None
    for i in range(attempts):
        text = prompt if i == 0 else (
            prompt + "\n\nIMPORTANT: answer ONLY by calling the provided function with all required fields. "
                     "No prose, no tables, no markdown.")
        try:
            return llm.invoke(text)
        except Exception as e:  # noqa: BLE001 - we re-raise anything that is not this specific failure
            if "tool_use_failed" not in str(e) and "did not call a tool" not in str(e):
                raise
            last_error = e
    raise last_error


def _log(step: str, status: str, summary: str) -> dict:
    return {"investigation_log": [{
        "step": step, "status": status, "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]}


def _checks_for(state: dict) -> dict:
    return compute_limit_checks(state.get("sensor_readings", []))


def _failure_reported(state: dict) -> bool:
    """Defaults to True: if the caller does not say, assume a failure is being investigated.
    Health-check mode only happens when the caller explicitly passes failure_reported=False."""
    value = state.get("failure_reported")
    return True if value is None else bool(value)


def _shorten(text: str, limit: int = 160) -> str:
    """Cut at a word boundary and say so, instead of stopping mid-sentence."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:-\u2011") + " ..."


def _maintenance_as_evidence() -> bool:
    """The maintenance records in this project are RANDOM templates (Phase 1), so by default the model is not shown
    them as evidence (they still appear in the report). Set MAINTENANCE_AS_EVIDENCE=1 to let it see them, e.g. for an
    ablation, or in a deployment with real records."""
    return os.getenv("MAINTENANCE_AS_EVIDENCE", "0") == "1"


def _derived_gap_mean(stats: dict):
    """Mean (process - air) temperature gap over the history, derived from the two channel means."""
    try:
        return stats["process_temperature"]["mean"] - stats["air_temperature"]["mean"]
    except (KeyError, TypeError):
        return None


def _facts_block(checks: dict) -> str:
    return "\n".join(f"- {f}" for f in limit_check_facts(checks))


# ─────────────────────────────────────────────
# 1. gather_context
# ─────────────────────────────────────────────
def gather_context(state: dict) -> dict:
    machine_id = state["machine_id"]
    as_of = state.get("as_of")
    sensor_data = get_sensor_data.invoke({"machine_id": machine_id, "limit": 20, "as_of": as_of})
    stats = calculate_sensor_statistics.invoke({"machine_id": machine_id, "as_of": as_of})

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
    result = run_anomaly_detection.invoke({"machine_id": state["machine_id"], "as_of": state.get("as_of")})
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

    as_of = state.get("as_of")
    maintenance = get_maintenance_history.invoke({"machine_id": machine_id, "as_of": as_of})
    history = get_failure_history.invoke({"machine_id": machine_id, "exclude_incident_id": exclude_id, "as_of": as_of})

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
    checks = _checks_for(state)
    health = is_health_check(anomaly, _failure_reported(state), checks)

    warning_line = ("WARNING: " + (anomaly.get("reliability_note") or "")) if anomaly.get("reliability_warning") else ""
    tie_line = (f"Close runner-up: {anomaly.get('close_second_class')} (margin {anomaly.get('close_second_margin')})"
                if near_tie(anomaly) else "No close runner-up class: the top prediction is not a near-tie.")
    catalog = "\n".join(f"- {c}" for c in FAULT_CATALOG)

    if health:
        task = """This is a HEALTH CHECK, not a failure investigation. No failure has been reported; the machine was
only flagged by a routine check. The classifier predicts no failure and no documented operating limit is
violated, so do not assume the routine check found a fault. Choose 1-2 labels from the allowed list that
deserve a second look, using only the numbers above. Do not include a 'no fault' option; the system adds it."""
    else:
        task = """Choose 2-4 labels from the allowed list. The ML prediction is a STARTING hypothesis, not a confirmed
answer - if it carries a reliability warning, treat it with appropriate skepticism and still consider other labels.
Do not include an 'unexplained' option; the system adds it."""

    prompt = f"""You are investigating an industrial machine.

Incident description: {state.get('incident_description') or '(none provided)'}

ML classifier prediction: {anomaly.get('predicted_class')} (confidence {anomaly.get('confidence')})
{warning_line}
{tie_line}

COMPUTED LIMIT CHECKS (calculated in code from the sensor readings - rely on these, do not recompute):
{_facts_block(checks)}

Recent maintenance: {len(state.get('maintenance_records', []))} records on file (synthetic).
Past failures on this machine: {len(state.get('historical_incidents', []))} incidents on file.

ALLOWED CANDIDATE CAUSES (the only failure modes this system's data and manuals can test):
{catalog}

{task}

Rules: set "cause" to EXACTLY one label from the allowed list and put all explanation in "rationale". Do not
invent other causes. The readings are independent samples with synthetic timestamps, so there is no real time
ordering: do not reason about trends."""

    result: HypothesisList = _invoke_structured(HypothesisList, prompt)
    hypotheses, notes = finalize_hypotheses([h.model_dump() for h in result.hypotheses], anomaly, checks, health,
                                            _failure_reported(state))

    mode = "health check" if health else "failure investigation"
    summary = f"Mode: {mode}. Hypotheses: " + ", ".join(h["cause"] for h in hypotheses)
    if notes:
        summary += " | " + "; ".join(notes)
    return {"hypotheses": hypotheses, **_log("generate_hypotheses", "done", summary)}


# ─────────────────────────────────────────────
# 5. investigate_hypotheses
# ─────────────────────────────────────────────
def investigate_hypotheses(state: dict) -> dict:
    anomaly = state.get("anomaly_result", {})
    checks = _checks_for(state)
    health = is_health_check(anomaly, _failure_reported(state), checks)

    if _maintenance_as_evidence():
        maintenance_text = "\n".join(
            f"- {r['date']}: {r['maintenance_type']} - {r['description']}"
            for r in state.get("maintenance_records", [])
        ) or "(none)"
    else:
        maintenance_text = ("(not provided: the maintenance records in this system are randomly generated demo data "
                            "and carry no information about this machine's condition)")

    similar_text = "\n".join(
        f"- Incident {r['incident_id']} ({r['machine_id']}, {r['date']}): {r['failure_type']} "
        f"[sensor distance: {r.get('sensor_distance')}]"
        for r in state.get("similar_incidents", [])
    ) or "(none)"

    # Trend labels are left out on purpose: readings are independent samples with synthetic timestamps.
    stats = state.get("sensor_statistics", {}).get("channels", {})
    stats_text = "\n".join(
        f"- {ch}: " + ", ".join(f"{k}={v}" for k, v in s.items() if k != "trend") for ch, s in stats.items()
    ) or "(none)"
    gap = _derived_gap_mean(stats)
    if gap is not None:
        stats_text += f"\n- derived: mean temperature gap (process minus air) over this history = {gap:.2f} K"

    evaluations, cap_notes, n_rule = [], [], 0
    for hyp in state.get("hypotheses", []):
        if health and hyp["cause"] == H_NO_FAULT:
            continue  # the health-check null hypothesis is evaluated by code below
        basis = None
        if not is_null_hypothesis(hyp["cause"]):
            rule_conf, why = rule_confidence(hyp["cause"], checks, anomaly, _failure_reported(state))
            basis = why
            if rule_conf == "low":
                # The documented rules already settle this one and nothing the model says can raise it, so it is not
                # sent to the model at all: that saves most of the tokens of a case.
                evaluations.append({
                    "cause": hyp["cause"], "supporting_evidence": [],
                    "contradicting_evidence": [why[0].upper() + why[1:] + "."],
                    "reasoning": f"Not sent to the model (the documented rules already settle it): {why}.",
                    "confidence": "low", "confidence_note": f"set to low by rule: {why}", "evaluated_by": "rule",
                    "confidence_basis": why})
                n_rule += 1
                continue
        docs = search_technical_documents.invoke({"query": hyp["cause"], "k": 3})
        docs_text = "\n".join(
            f"- [{d['source']} > {d['section']}] {d['content'][:400]}"
            for d in docs.get("results", [])
        ) or "(no relevant documentation found)"

        prompt = f"""Evaluate this candidate root cause against the evidence below. Be skeptical -
only cite something as supporting or contradicting evidence if it specifically relates to this cause.

CANDIDATE CAUSE: {hyp['cause']}
Rationale given: {hyp['rationale']}

COMPUTED LIMIT CHECKS (calculated in code - use these numbers for every threshold comparison, never recompute):
{_facts_block(checks)}

SENSOR STATISTICS over this machine's FULL reading history up to now (not just the 20 recent readings; no
trend information: the readings are independent samples with synthetic timestamps):
{stats_text}

TECHNICAL DOCUMENTATION (retrieved for this specific cause; synthetic manuals):
{docs_text}

MAINTENANCE HISTORY (synthetic; a MISSING record is not evidence of anything):
{maintenance_text}

SIMILAR PAST INCIDENTS (ranked by sensor-profile similarity). These are drawn ONLY from past failures, so
resemblance to a past failure is NOT evidence that this machine is failing:
{similar_text}

Give your own honest confidence rating (low / medium / high). The system compares it with the documented limits
and may override it; both ratings are kept. Your main job is the evidence: list what specifically supports and what
specifically contradicts this cause, using the computed checks for every threshold.
Evaluate: what supports this cause, what contradicts it, and how confident should we be?"""

        result: HypothesisEvaluation = _invoke_structured(HypothesisEvaluation, prompt)
        ev = result.model_dump()

        ev["llm_confidence"] = ev["confidence"]
        if basis:
            ev["confidence_basis"] = basis
        new_conf, note = decide_confidence(ev["cause"], ev["confidence"], checks, anomaly, _failure_reported(state))
        if note:
            ev["confidence_note"] = note
            ev["confidence"] = new_conf
            ev["reasoning"] = f"{ev['reasoning']} [{note}]"
            cap_notes.append(f"'{ev['cause']}' {note}")
        evaluations.append(ev)

    if health:
        top = max((ORDER.index(e["confidence"]) for e in evaluations), default=0)
        conf = anomaly.get("confidence")
        conf_txt = f"{conf:.4f}" if isinstance(conf, (int, float)) else "unknown"
        evaluations.append({
            "cause": H_NO_FAULT,
            "supporting_evidence": [
                f"The classifier predicts {anomaly.get('predicted_class')} (confidence {conf_txt}).",
                f"No documented operating limit is violated in the {checks['n_readings']} recent readings.",
            ],
            "contradicting_evidence": [f"'{e['cause']}' was rated {e['confidence']}."
                                       for e in evaluations if e["confidence"] != "low"],
            "reasoning": "Set by code: 'medium' when every fault hypothesis stays at low confidence, otherwise 'low'. "
                         "An absence of evidence of a fault, not proof that none exists.",
            "confidence": "medium" if top == 0 else "low",
        })

    summary = ", ".join(f"{e['cause']} ({e['confidence']})" for e in evaluations)
    if n_rule:
        summary += f" | {n_rule} settled by rule without the model"
    if cap_notes:
        summary += " | RULE OVERRIDES: " + "; ".join(cap_notes)

    return {
        "hypothesis_evaluations": evaluations,
        **_log("investigate_hypotheses", "done", f"Evaluated {len(evaluations)} hypotheses: {summary}"),
    }


def route_after_investigation(state: dict) -> str:
    """Conditional edge: loop back for more evidence, or move on to a conclusion.
    Retrying only widens the similar-incident list, so it is only worth doing when some
    hypothesis is 'medium' (promising but unproven). If everything is low, more of the same
    evidence will not change the answer."""
    evaluations = state.get("hypothesis_evaluations", [])
    confidences = [e["confidence"] for e in evaluations]
    promising = [e["confidence"] for e in evaluations if not is_null_hypothesis(e["cause"])]
    retry_count = state.get("retry_count", 0)

    # A 'medium' on a null hypothesis ("No fault" / "Unexplained") is not worth chasing.
    if "high" in confidences or retry_count >= MAX_RETRIES or "medium" not in promising:
        return "sufficient"
    return "insufficient"


# ─────────────────────────────────────────────
# 6. gather_more_evidence (only runs on the "insufficient" branch)
# ─────────────────────────────────────────────
def gather_more_evidence(state: dict) -> dict:
    """Widen the similar-incident search, then loop back to investigate_hypotheses.
    HONEST LIMIT: this only adds more of the same kind of evidence (more failure look-alikes)."""
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
def _evidence_text(state: dict, checks: dict) -> str:
    gap = _derived_gap_mean(state.get("sensor_statistics", {}).get("channels", {}))
    return "\n".join([
        f"derived mean temperature gap over history {gap:.2f}" if gap is not None else "",
        str(state.get("sensor_statistics", {}).get("channels", {})),
        str(state.get("sensor_readings", [])),
        "\n".join(map(str, state.get("maintenance_records", []))),
        "\n".join(map(str, state.get("similar_incidents", []))),
        "\n".join(map(str, state.get("historical_incidents", []))),
        str(state.get("anomaly_result", {})),
        "\n".join(limit_check_facts(checks)),
        DOC_CONSTANTS_TEXT,
    ])


def _ml_prediction_lines(anomaly: dict) -> list:
    conf = anomaly.get("confidence")
    conf_txt = f"{conf:.4f}" if isinstance(conf, (int, float)) else "unknown"
    lines = [f"PREDICTION: {anomaly.get('predicted_class')} (confidence {conf_txt}) - "
             f"reliability_warning: {'YES - ' + str(anomaly.get('reliability_note')) if anomaly.get('reliability_warning') else 'none'}"]
    if near_tie(anomaly):
        lines.append(f"Close runner-up: {anomaly.get('close_second_class')} "
                     f"(margin {anomaly.get('close_second_margin')}) - the model is uncertain between these.")
    return lines


NO_FAULT_TEXT = "No fault identified (ML predicts No Failure; no documented limit violated)"


def determine_root_cause(state: dict) -> dict:
    """The CODE decides the outcome from the evaluated confidences (which documented rules have already
    settled); the LLM only writes the justification and the observed facts for that outcome."""
    evaluations = state.get("hypothesis_evaluations", [])
    anomaly = state.get("anomaly_result", {})
    checks = _checks_for(state)
    reported = _failure_reported(state)
    health = is_health_check(anomaly, reported, checks)
    advisory = tool_wear_advisory(checks, anomaly, reported)

    decision = decide_final(evaluations, health)
    if decision["kind"] == "causes":
        decided = " + ".join(decision["causes"])
    elif decision["kind"] == "no_fault":
        decided = NO_FAULT_TEXT
    elif decision["kind"] == "unexplained":
        decided = decision["causes"][0]
    else:
        decided = "Unresolved - insufficient evidence"

    evaluations_text = "\n\n".join(
        f"Cause: {e['cause']}\nConfidence: {e['confidence']}\n"
        f"Supporting: {e['supporting_evidence']}\nContradicting: {e['contradicting_evidence']}\n"
        f"Reasoning: {e['reasoning']}"
        for e in evaluations
    ) or "(no hypotheses were evaluated)"

    prompt = f"""An investigation of an industrial machine has finished evaluating its hypotheses.

OUTCOME DECIDED BY THE SYSTEM from the evaluations and the documented limits: {decided} (confidence: {decision['confidence']}).
Write the justification for this outcome and list the observed facts. Do not change the outcome.

HYPOTHESIS EVALUATIONS:
{evaluations_text}

Do NOT discuss the confidence level or why it is not higher: the system adds the documented basis itself.

COMPUTED LIMIT CHECKS (calculated in code - facts):
{_facts_block(checks)}

ML CLASSIFIER CONTEXT (a prediction, not a confirmed fact):
predicted_class={anomaly.get('predicted_class')}, confidence={anomaly.get('confidence')}
reliability_warning={anomaly.get('reliability_warning')}: {anomaly.get('reliability_note')}

Separate observed facts (directly from sensor/maintenance/incident records or the computed checks) from ML
predictions from unsupported speculation. Never write a number that is not in the material above.
Observed facts must be plain facts with NO interpretation (no "indicating...", "suggesting..."). Do not repeat the
computed checks, and never say a value is "within range" by comparing it with the min/max of the same history:
that is always true. If the outcome is weakly supported, say so plainly."""

    result: RootCauseDetermination = _invoke_structured(RootCauseDetermination, prompt)
    root = result.model_dump()
    root["most_supported_cause"] = decided
    root["confidence_level"] = decision["confidence"]
    notes = [f"outcome decided by code ({decision['kind']})"]

    non_null = [e for e in evaluations if not is_null_hypothesis(e["cause"])]
    if decision["kind"] == "no_fault":
        root["justification"] = (
            f"The classifier predicts {anomaly.get('predicted_class')} and no documented operating limit is violated by "
            f"the latest reading. None of the {len(non_null)} candidate faults reached more than low confidence. This "
            f"is an absence of evidence of a fault, not proof that none exists; the timestamps, maintenance records and "
            f"manuals are synthetic, which limits what can be concluded." + (f" ADVISORY: {advisory}" if advisory else ""))
    elif decision["kind"] == "causes":
        # the real reason for each confidence is written by code: the model's own account of it was unreliable
        basis = "; ".join(f"{e['cause']} - {e['confidence_basis']}" for e in evaluations
                          if e["cause"] in decision["causes"] and e.get("confidence_basis"))
        if basis:
            root["justification"] = f"{str(root.get('justification')).rstrip()} Confidence basis (set by documented rules): {basis}."
        if not ml_agrees(decision["causes"], anomaly):
            conf = anomaly.get("confidence")
            conf_txt = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
            root["justification"] += (f" Classifier disagreement: it predicts {anomaly.get('predicted_class')}{conf_txt}, "
                                      f"which does not name {'this cause' if len(decision['causes']) == 1 else 'these causes'}.")
    elif decision["kind"] == "unexplained":
        root["justification"] = (
            "Every specific fault hypothesis stayed at low confidence and no documented limit is violated, so the "
            "'unexplained' reading is the best fit. That reflects an absence of evidence, not a diagnosed cause. "
            "Model notes (NOT a conclusion): " + str(root.get("justification")))
    elif decision["kind"] == "unresolved":
        root["justification"] = ("No candidate cause reached more than low confidence. Model notes (NOT a conclusion): "
                                 + str(root.get("justification")))
    root["advisories"] = [advisory] if advisory else []

    # 'unsupported_speculation' is built by code from the hypotheses that were ACTUALLY evaluated and not chosen.
    chosen = set(decision["causes"])
    rejected = []
    for e in evaluations:
        if e["cause"] in chosen or (e["cause"] == H_NO_FAULT and decision["kind"] == "no_fault"):
            continue
        against = (e.get("contradicting_evidence") or ["no direct supporting evidence found"])[0]
        rejected.append(f"{e['cause']} (confidence {e['confidence']}): {_shorten(against)}")
    root["unsupported_speculation"] = rejected

    # ML predictions are written by code, so the runner-up class cannot be mis-described.
    root["ml_predictions"] = _ml_prediction_lines(anomaly)

    # Observed facts: computed facts first, then the LLM's, with unverifiable numbers tagged.
    llm_facts, flagged = annotate_ungrounded(root.get("observed_facts", []), _evidence_text(state, checks))
    llm_facts = drop_redundant_facts(llm_facts, "\n".join(limit_check_facts(checks)))
    root["observed_facts"] = [f"[computed] {f}" for f in limit_check_facts(checks)] + llm_facts
    if flagged:
        notes.append(f"{flagged} observed fact(s) contain numbers not found in the evidence")

    summary = f"Most supported cause: {root['most_supported_cause']} ({root['confidence_level']}) | " + "; ".join(notes)
    return {"root_cause": root, **_log("determine_root_cause", "done", summary)}


# ─────────────────────────────────────────────
# 8. generate_report
# ─────────────────────────────────────────────
def generate_report(state: dict) -> dict:
    """Programmatic assembly of already-structured data - no extra LLM call, so the report
    cannot drift from the validated structured facts."""
    anomaly = state.get("anomaly_result", {})
    root_cause = state.get("root_cause", {})
    checks = _checks_for(state)

    evidence_sources = []
    for r in state.get("similar_incidents", []):
        evidence_sources.append(f"Incident {r['incident_id']} ({r['machine_id']}, {r['date']})")

    report = {
        "incident_summary": state.get("incident_description", "(no description provided)"),
        "mode": "health_check" if is_health_check(anomaly, _failure_reported(state), checks)
                else "failure_investigation",
        "machine_condition": {
            "machine_id": state["machine_id"],
            "latest_reading": state["sensor_readings"][0] if state.get("sensor_readings") else None,
            # trend labels are left out: the readings are independent samples, so a 'trend' is not a real trend
            "sensor_statistics": {ch: {k: v for k, v in st.items() if k != "trend"}
                                  for ch, st in state.get("sensor_statistics", {}).get("channels", {}).items()},
        },
        "limit_checks": checks,
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
        "advisories": root_cause.get("advisories", []),
        "unsupported_speculation": root_cause.get("unsupported_speculation", []),
        "evidence_sources": evidence_sources,
        "investigation_log": state.get("investigation_log", []),
    }

    return {
        "final_report": report,
        **_log("generate_report", "done", "Report assembled."),
    }
