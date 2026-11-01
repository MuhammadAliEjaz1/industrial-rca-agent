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
    "tool wear 200 240 minutes; proactive tool replacement 180 200 minutes; random failure about 0.1 %"
)

# Closed hypothesis catalog: only failure modes this system can test with its data, limits and manuals.
H_TOOL_WEAR = "Tool wear failure"
H_HEAT = "Heat dissipation failure"
H_POWER = "Power failure"
H_OVERSTRAIN = "Overstrain failure"
H_SENSOR = "Sensor fault or data-quality problem"
H_UNEXPLAINED = "Unexplained: random or undetected failure"   # failure reported, nothing supported
H_NO_FAULT = "No fault: normal variation"                       # health check, nothing reported
FAULT_CATALOG = [H_TOOL_WEAR, H_HEAT, H_POWER, H_OVERSTRAIN, H_SENSOR]

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
    # 'violated' looks at the LATEST reading only: the readings in the window are independent samples, so an
    # older row that broke a limit says nothing about the present (the counts above are shown for context).
    violated = [k for k in ("heat_dissipation", "power_out_of_range", "possible_overstrain") if per[0][k]]
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
        f"Note: the {n} readings loaded for this machine are independent samples with synthetic timestamps, not a "
        f"time series. Decisions use the latest reading only.",
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
    return (not failure_reported and anomaly.get("predicted_class") == "No Failure"
            and not checks.get("violated"))


# ---------- confidence: decided by documented rules, not by the LLM ----------
# The first pilot showed the LLM overriding documented rules (heat dissipation rated 'low' although the limit was
# met) and rating tool wear 'high' on readings that broke a different limit. So wherever a documented criterion
# exists, the code sets the confidence; the LLM's own rating is kept next to it as a second opinion.

RULE_KEY = {H_HEAT: "heat_dissipation", H_POWER: "power_out_of_range", H_OVERSTRAIN: "possible_overstrain"}
_ML_WORD = {H_HEAT: "heat dissipation", H_POWER: "power", H_OVERSTRAIN: "overstrain", H_TOOL_WEAR: "tool wear"}


def is_null_hypothesis(cause: str) -> bool:
    return cause.strip().lower().startswith(NULL_PREFIXES)


def rule_confidence(cause: str, checks: dict, anomaly: dict, failure_reported: bool):
    """(confidence, why) that the documented rules give a NON-null cause. Independent of anything the LLM says."""
    latest = checks.get("latest") or {}
    ml_class = str(anomaly.get("predicted_class") or "").lower()
    ml_p = anomaly.get("confidence") or 0
    ml_names = cause in _ML_WORD and _ML_WORD[cause] in ml_class and ml_p >= 0.5

    if cause in RULE_KEY:
        if latest.get(RULE_KEY[cause]):
            return "high", "its documented limit is met by the latest reading"
        if ml_names:
            return "medium", "its documented limit is not met by the latest reading, but the classifier names it"
        return "low", "its documented limit is not met by the latest reading"
    if cause == H_TOOL_WEAR:
        wear = bool(latest.get("tool_wear_at_failure_range"))
        if checks.get("violated"):
            return "low", "a documented hard limit already explains the reading, and tool wear alone is a weak signal"
        if (failure_reported and wear) or ml_names:
            return "medium", ("tool wear can reach 'medium' at most: it is a weak signal "
                              "(documented recall ~0.12; about 94% of readings at 200+ min are healthy)")
        if wear:
            return "low", "no failure was reported, so a worn tool is an advisory, not a diagnosis"
        return "low", "the latest tool wear is below the documented failure range"
    return "low", "a sensor fault cannot be established from this data (no independent cross-check)"


def decide_confidence(cause: str, llm_conf: str, checks: dict, anomaly: dict, failure_reported: bool):
    """Returns (confidence, note). note is None when the LLM's rating is kept (null hypotheses) or agrees."""
    if is_null_hypothesis(cause):
        return llm_conf, None
    new, why = rule_confidence(cause, checks, anomaly, failure_reported)
    note = None if new == llm_conf else f"set to {new} by rule (the model said {llm_conf}): {why}"
    return new, note


def ml_agrees(causes: list, anomaly: dict) -> bool:
    """True if the classifier's top class names at least one of the chosen causes."""
    ml_class = str(anomaly.get("predicted_class") or "").lower()
    return any(_ML_WORD.get(c) and _ML_WORD[c] in ml_class for c in causes)


def decide_final(evaluations: list, health: bool) -> dict:
    """The outcome, from the evaluated confidences. kind: causes | no_fault | unexplained | unresolved."""
    non_null = [e for e in evaluations if not is_null_hypothesis(e["cause"])]
    rank = {c: i for i, c in enumerate(FAULT_CATALOG)}
    for level in ("high", "medium"):
        hits = sorted((e["cause"] for e in non_null if e["confidence"] == level), key=lambda c: rank.get(c, 99))
        if hits:
            return {"kind": "causes", "causes": hits, "confidence": level}
    if health:
        return {"kind": "no_fault", "causes": [], "confidence": "medium"}
    nulls = [e for e in evaluations if is_null_hypothesis(e["cause"]) and e["confidence"] in ("medium", "high")]
    if nulls:
        best = max(nulls, key=lambda e: ORDER.index(e["confidence"]))
        return {"kind": "unexplained", "causes": [best["cause"]], "confidence": "medium"}
    return {"kind": "unresolved", "causes": [], "confidence": "low"}


def tool_wear_advisory(checks: dict, anomaly: dict, failure_reported: bool):
    """A routine check with tool wear in the failure range and nothing else wrong gets an advisory, not a diagnosis."""
    latest = checks.get("latest") or {}
    if (not failure_reported and latest.get("tool_wear_at_failure_range") and not checks.get("violated")
            and anomaly.get("predicted_class") == "No Failure"):
        return (f"Latest tool wear is {checks['latest_raw']['tool_wear']:g} min, inside the documented failure range "
                f"(about {TOOL_WEAR_FAIL_START}-240 min). Sensors cannot reliably detect tool-wear failures (documented "
                f"recall ~0.12; about 94% of readings at 200+ min are healthy), so this is a maintenance advisory, not a "
                f"diagnosis. The manual recommends proactive replacement at about 180-200 min.")
    return None


# ---------- number grounding ----------

_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _clean(text: str) -> str:
    """Joins thousands written with a thin/no-break space ('12\u202f337' -> '12337') so they are one number."""
    return re.sub(r"(\d)[\u202f\u00a0](?=\d{3}\b)", r"\1", text).replace("\u202f", " ")


def _to_float(s: str):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def ungrounded_numbers(claim: str, evidence_text: str) -> list:
    """Numbers in `claim` (decimals or 3+ digits) that match nothing in the evidence (0.5% tolerance)."""
    clean = _clean
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


# ---------- closed hypothesis catalog ----------
# Only failure modes this system can actually test with its data (sensor limits, ML class, manuals).
# Power sags, PLC interlocks, bearing seizures etc. cannot be confirmed or ruled out here, so they are not offered.


_SHORTCUTS = [("tool wear", H_TOOL_WEAR), ("heat dissipation", H_HEAT), ("power", H_POWER),
              ("overstrain", H_OVERSTRAIN), ("sensor", H_SENSOR), ("unexplained", H_UNEXPLAINED),
              ("random", H_UNEXPLAINED), ("no fault", H_NO_FAULT)]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z]+", " ", text.lower()).strip()


def match_catalog(cause: str):
    """Catalog label for an LLM-written cause, or None if it is outside the catalog."""
    n = _norm(cause)
    for label in FAULT_CATALOG + [H_UNEXPLAINED, H_NO_FAULT]:
        if n == _norm(label):
            return label
    for key, label in _SHORTCUTS:
        if n == key or n.startswith(key + " "):
            return label
    return None


def required_hypotheses(anomaly: dict, checks: dict, failure_reported: bool = True) -> list:
    """(label, reason) pairs that are ALWAYS evaluated, whether or not the LLM proposed them."""
    req = {}
    violated = checks.get("violated", [])
    if "heat_dissipation" in violated:
        req[H_HEAT] = "the documented heat-dissipation rule is met in the recent readings"
    if "power_out_of_range" in violated:
        req[H_POWER] = "derived power is outside the documented range in the recent readings"
    if "possible_overstrain" in violated:
        req[H_OVERSTRAIN] = "wear x torque is above the lowest documented overstrain limit in the recent readings"
    if failure_reported and (checks.get("latest") or {}).get("tool_wear_at_failure_range"):
        req[H_TOOL_WEAR] = "the latest tool wear is in the documented failure range"
    cls = str(anomaly.get("predicted_class") or "")
    if cls.lower() != "no failure":
        for word, label in (("heat dissipation", H_HEAT), ("overstrain", H_OVERSTRAIN),
                            ("power", H_POWER), ("tool wear", H_TOOL_WEAR)):
            if word in cls.lower():
                req.setdefault(label, f"the classifier predicts {cls}")
    return list(req.items())


def finalize_hypotheses(llm_items: list, anomaly: dict, checks: dict, health: bool, failure_reported: bool = True):
    """Keeps only catalog hypotheses from the LLM, adds the ones the evidence forces, and appends the null
    hypothesis. Returns (hypotheses, notes)."""
    chosen, dropped, notes = {}, [], []
    for h in llm_items:
        label = match_catalog(h["cause"])
        if label is None:
            dropped.append(h["cause"])
        elif label not in (H_UNEXPLAINED, H_NO_FAULT):
            chosen.setdefault(label, h["rationale"])
    for label, why in required_hypotheses(anomaly, checks, failure_reported):
        if label not in chosen:
            chosen[label] = f"Added by rule: {why}."
            notes.append(f"added '{label}' because {why}")
    hyps = [{"cause": k, "rationale": v} for k, v in chosen.items()]
    if health:
        hyps.append({"cause": H_NO_FAULT, "rationale":
                     "No failure was reported, the classifier predicts no failure and no documented limit is "
                     "violated, so normal variation is the default explanation."})
    else:
        hyps.append({"cause": H_UNEXPLAINED, "rationale":
                     "The manual says to report a failure as unexplained when every mechanism has been checked and "
                     "none is supported (random failures, about 0.1%, leave no sensor signature)."})
    if dropped:
        notes.append("dropped (outside the documented failure modes): " + "; ".join(dropped))
    return hyps, notes


# ---------- duplicate facts ----------

def _significant_numbers(text: str) -> list:
    out = []
    for m in _NUM.findall(_clean(text)):
        digits = m.replace(",", "").rstrip(".")
        if "." in digits or len(digits) >= 3:
            v = _to_float(m)
            if v is not None:
                out.append(v)
    return out


def drop_redundant_facts(facts: list, computed_text: str) -> list:
    """Drops facts whose every significant number already appears in the computed facts (they only repeat them)."""
    known = _significant_numbers(computed_text)
    keep = []
    for f in facts:
        nums = _significant_numbers(f)
        if nums and all(any(abs(v - k) <= max(0.005 * abs(k), 0.05) for k in known) for v in nums):
            continue
        keep.append(f)
    return keep
