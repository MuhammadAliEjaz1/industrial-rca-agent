"""
Deterministic checks for the investigation agent (NO LLM, no heavy imports, easy to test).

Why this file exists: the first real Groq runs showed the LLM doing arithmetic itself
(it reported a 11.3 K temperature gap when the data gave 9.7 K), reading the heat-dissipation
threshold backwards, and rating the same evidence low/medium/high on different runs.
Anything that is a simple rule from the manuals is now computed here, in code.

All thresholds come from the SYNTHETIC manuals in rag/knowledge_base/.
"""

import math
import re

HEAT_DIFF_K = 8.6          # heat dissipation failure: (process - air) < 8.6 K ...
HEAT_RPM = 1380            # ... AND rotational speed < 1380 rpm
POWER_MIN_W = 3500         # power failure: derived power outside 3,500 - 9,000 W
POWER_MAX_W = 9000
OVERSTRAIN_LIMITS = {"L": 11000, "M": 12000, "H": 13000}   # tool_wear * torque, by product quality
TOOL_WEAR_FAIL_START = 200  # tool wear failures happen around 200-240 min

# Plain text of the documented constants. Used only to decide whether a number in the
# agent's "observed facts" appears anywhere in the evidence.
DOC_CONSTANTS_TEXT = (
    "heat dissipation 8.6 K 1380 rpm; power 3500 W 9000 W; overstrain 11000 12000 13000; "
    "tool wear 200 240 minutes; random failure about 0.1 %"
)

ORDER = ["low", "medium", "high"]
NULL_PREFIXES = ("no fault", "unexplained", "normal variation")


# ---------- per-reading rules ----------

def check_reading(r: dict) -> dict:
    temp_diff = r["process_temperature"] - r["air_temperature"]
    power_w = r["torque"] * r["rotational_speed"] * 2 * math.pi / 60
    wear_x_torque = r["tool_wear"] * r["torque"]
    return {
        "temp_diff_k": temp_diff,
        "power_w": power_w,
        "wear_x_torque": wear_x_torque,
        "heat_dissipation": temp_diff < HEAT_DIFF_K and r["rotational_speed"] < HEAT_RPM,
        "power_out_of_range": power_w < POWER_MIN_W or power_w > POWER_MAX_W,
        # Product quality may not be known, so we flag against the LOWEST tier (L = 11,000).
        # This can over-flag for M/H quality products. Treat as "possible overstrain".
        "possible_overstrain": wear_x_torque > OVERSTRAIN_LIMITS["L"],
        "tool_wear_at_failure_range": r["tool_wear"] >= TOOL_WEAR_FAIL_START,
    }


def compute_limit_checks(readings: list) -> dict:
    """readings: newest first (index 0 = latest), each with the 5 sensor values."""
    if not readings:
        return {"n_readings": 0, "latest": None, "latest_raw": None, "counts": {}, "violated": []}
    per = [check_reading(r) for r in readings]
    counts = {k: sum(1 for p in per if p[k])
              for k in ("heat_dissipation", "power_out_of_range", "possible_overstrain",
                        "tool_wear_at_failure_range")}
    # Tool wear >= 200 is only a SIGNAL (671 normal readings overlap that range), not a hard limit.
    violated = [k for k in ("heat_dissipation", "power_out_of_range", "possible_overstrain") if counts[k] > 0]
    return {"n_readings": len(readings), "latest": per[0], "latest_raw": readings[0],
            "counts": counts, "violated": violated}


def limit_check_facts(checks: dict) -> list:
    if not checks.get("latest"):
        return ["No sensor readings available, so no limit checks could be computed."]
    L, raw, c, n = checks["latest"], checks["latest_raw"], checks["counts"], checks["n_readings"]
    return [
        f"Latest reading: process minus air temperature = {L['temp_diff_k']:.1f} K at {raw['rotational_speed']:g} rpm. "
        f"Heat-dissipation failure needs the gap below {HEAT_DIFF_K} K AND speed below {HEAT_RPM} rpm "
        f"(met: {'yes' if L['heat_dissipation'] else 'no'}).",
        f"Latest derived power = {L['power_w']:.0f} W (documented normal range {POWER_MIN_W}-{POWER_MAX_W} W; "
        f"outside range: {'yes' if L['power_out_of_range'] else 'no'}).",
        f"Latest tool_wear x torque = {L['wear_x_torque']:.0f} (overstrain limits by product quality: "
        f"L {OVERSTRAIN_LIMITS['L']}, M {OVERSTRAIN_LIMITS['M']}, H {OVERSTRAIN_LIMITS['H']}; "
        f"above the lowest limit: {'yes' if L['possible_overstrain'] else 'no'}).",
        f"Latest tool wear = {raw['tool_wear']:g} min (tool-wear failures are documented from about "
        f"{TOOL_WEAR_FAIL_START} min; at or above: {'yes' if L['tool_wear_at_failure_range'] else 'no'}).",
        f"Across the {n} most recent readings: heat-dissipation rule met in {c['heat_dissipation']}, "
        f"power out of range in {c['power_out_of_range']}, wear x torque above the lowest limit in "
        f"{c['possible_overstrain']}, tool wear at or above {TOOL_WEAR_FAIL_START} in {c['tool_wear_at_failure_range']}.",
    ]


# ---------- ML helpers ----------

def near_tie(anomaly: dict, max_margin: float = 0.25) -> bool:
    """True only when a runner-up class is genuinely close.
    ASSUMES close_second_margin = top probability minus second probability (small = close)."""
    m = anomaly.get("close_second_margin")
    return bool(anomaly.get("close_second_class")) and isinstance(m, (int, float)) and m <= max_margin


def is_health_check(anomaly: dict, failure_reported: bool, checks: dict) -> bool:
    """Nobody reported a failure, ML says No Failure, and no documented limit is violated.
    (If a failure WAS reported, e.g. a random failure with normal sensors, it is always an investigation.)"""
    # Tool wear at or above ~200 min is a documented failure range, and the classifier is known to miss
    # tool-wear failures (recall ~0.12), so that reading is never called a 'health check'.
    wear_signal = bool((checks.get("latest") or {}).get("tool_wear_at_failure_range"))
    return (not failure_reported and anomaly.get("predicted_class") == "No Failure"
            and not checks.get("violated") and not wear_signal)


# ---------- confidence caps ----------

def is_null_hypothesis(cause: str) -> bool:
    return cause.strip().lower().startswith(NULL_PREFIXES)


def cap_confidence(cause: str, confidence: str, checks: dict, anomaly: dict):
    """Returns (confidence, note). 'high' needs a documented limit violation.
    Null hypotheses ('No fault:', 'Unexplained:') are not capped."""
    if is_null_hypothesis(cause):
        return confidence, None
    cap, why = "high", ""
    if not checks.get("violated"):
        cap, why = "medium", "no documented operating limit is violated in the recent readings"
        p = anomaly.get("confidence") or 0
        wear_signal = bool((checks.get("latest") or {}).get("tool_wear_at_failure_range"))
        # With tool wear in the failure range we do not squeeze everything to 'low': the classifier
        # is documented as unreliable for tool wear, so a confident 'No Failure' means less there.
        if anomaly.get("predicted_class") == "No Failure" and p >= 0.95 and not wear_signal:
            cap = "low"
            why = "the classifier predicts No Failure with high confidence and no documented limit is violated"
    if ORDER.index(confidence) > ORDER.index(cap):
        return cap, f"capped from {confidence} to {cap}: {why}"
    return confidence, None


# ---------- number grounding ----------

_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _to_float(s: str):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def ungrounded_numbers(claim: str, evidence_text: str) -> list:
    """Numbers in `claim` (decimals or 3+ digits) that match nothing in the evidence (0.5% tolerance)."""
    clean = lambda t: t.replace("\u202f", " ")
    known = [v for v in (_to_float(m) for m in _NUM.findall(clean(evidence_text))) if v is not None]
    bad = []
    for m in _NUM.findall(clean(claim)):
        digits = m.replace(",", "").rstrip(".")
        if "." not in digits and len(digits) < 3:
            continue
        v = _to_float(m)
        if v is None:
            continue
        if not any(abs(v - k) <= max(0.005 * abs(k), 0.05) for k in known):
            bad.append(digits)
    return bad


def annotate_ungrounded(facts: list, evidence_text: str):
    """Tags facts containing numbers not found in the evidence. Returns (facts, flagged_count)."""
    out, flagged = [], 0
    for f in facts:
        bad = ungrounded_numbers(f, evidence_text)
        if bad:
            flagged += 1
            out.append(f"{f}  [UNVERIFIED NUMBER(S): {', '.join(bad)}]")
        else:
            out.append(f)
    return out, flagged
