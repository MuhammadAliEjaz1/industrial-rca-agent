"""Unit tests for graph/checks.py (no LLM, no database). From backend/:  python test_checks.py"""
from graph.checks import *

# ---- the real M001 reading from the first live run (torque 38 is a stand-in: it was not in the output) ----
latest = dict(air_temperature=298.6, process_temperature=308.3, rotational_speed=1589, torque=38.0, tool_wear=174)
older = dict(air_temperature=299.9, process_temperature=309.9, rotational_speed=1500, torque=40.0, tool_wear=100)
hot = dict(air_temperature=300.0, process_temperature=308.0, rotational_speed=1300, torque=40.0, tool_wear=50)
worn = dict(air_temperature=298.0, process_temperature=308.3, rotational_speed=1500, torque=30.0, tool_wear=215)

c = compute_limit_checks([latest, older])
assert round(c["latest"]["temp_diff_k"], 1) == 9.7 and c["violated"] == []
anom = {"predicted_class": "No Failure", "confidence": 0.999775, "close_second_class": "Power Failure", "close_second_margin": 0.9995}
assert is_health_check(anom, False, c) and not is_health_check(anom, True, c)
assert not near_tie(anom) and near_tie({"close_second_class": "X", "close_second_margin": 0.1})

# violations are judged on the LATEST reading: an older row that broke a limit must not count
c_old_hot = compute_limit_checks([latest, hot])
assert c_old_hot["violated"] == [] and c_old_hot["counts"]["heat_dissipation"] == 1
assert is_health_check(anom, False, c_old_hot)
ch = compute_limit_checks([hot, latest])
assert ch["violated"] == ["heat_dissipation"]
assert not is_health_check({"predicted_class": "Heat Dissipation Failure"}, False, ch)

# tool wear in the failure range: a signal, never a hard violation
cw = compute_limit_checks([worn])
assert cw["violated"] == [] and cw["latest"]["tool_wear_at_failure_range"]
assert is_health_check(anom, False, cw)                         # routine check: still a health check ...
adv = tool_wear_advisory(cw, anom, False)
assert adv and "215" in adv and "advisory" in adv               # ... with an advisory
assert tool_wear_advisory(cw, anom, True) is None               # a reported failure is investigated, not advised
assert tool_wear_advisory(c, anom, False) is None               # 174 min: nothing to say
assert tool_wear_advisory(ch, {"predicted_class": "Heat Dissipation Failure"}, False) is None

# ---- decide_confidence: documented rules set the confidence, the LLM's rating is only a second opinion ----
conf, note = decide_confidence(H_HEAT, "low", ch, {"predicted_class": "Heat Dissipation Failure", "confidence": .97}, True)
assert conf == "high" and "model said low" in note              # the pilot's inc-117 failure: rule met, LLM said low
assert decide_confidence(H_HEAT, "high", ch, {}, True) == ("high", None)
assert decide_confidence(H_HEAT, "high", c, anom, True)[0] == "low"              # rule not met, classifier silent
assert decide_confidence(H_POWER, "high", c, {"predicted_class": "Power Failure", "confidence": .9}, True)[0] == "medium"
# tool wear can never be 'high', and a hard limit already explains the reading -> 'low'
assert decide_confidence(H_TOOL_WEAR, "high", ch, anom, True)[0] == "low"
assert decide_confidence(H_TOOL_WEAR, "high", cw, anom, True)[0] == "medium"     # failure reported + wear >= 200
assert decide_confidence(H_TOOL_WEAR, "high", cw, anom, False)[0] == "low"       # routine check: advisory only
assert decide_confidence(H_TOOL_WEAR, "high", c, {"predicted_class": "Tool Wear Failure", "confidence": .6}, True)[0] == "medium"
assert decide_confidence(H_TOOL_WEAR, "high", c, anom, True)[0] == "low"         # 174 min, nothing else
assert decide_confidence(H_SENSOR, "high", ch, anom, True)[0] == "low"           # a sensor fault cannot be established
assert decide_confidence(H_UNEXPLAINED, "medium", c, anom, True) == ("medium", None)
assert decide_confidence(H_NO_FAULT, "high", c, anom, False) == ("high", None)

# ---- does the classifier name the chosen cause? ----
assert ml_agrees([H_HEAT], {"predicted_class": "Heat Dissipation Failure + Overstrain Failure"})
assert ml_agrees([H_POWER, H_OVERSTRAIN], {"predicted_class": "Power Failure"})
assert not ml_agrees([H_TOOL_WEAR], {"predicted_class": "No Failure", "confidence": 0.91})
assert not ml_agrees([H_HEAT], {"predicted_class": "Multiple Failure Causes (rare combination)"})

# ---- decide_final ----
ev = lambda **k: [{"cause": a, "confidence": b} for a, b in k.items()]
E = lambda pairs: [{"cause": a, "confidence": b} for a, b in pairs]
assert decide_final(E([(H_OVERSTRAIN, "high"), (H_POWER, "high"), (H_TOOL_WEAR, "low"), (H_UNEXPLAINED, "low")]), False) == \
    {"kind": "causes", "causes": [H_POWER, H_OVERSTRAIN], "confidence": "high"}     # compound, catalog order
assert decide_final(E([(H_TOOL_WEAR, "medium"), (H_SENSOR, "low")]), False)["confidence"] == "medium"
assert decide_final(E([(H_POWER, "high"), (H_TOOL_WEAR, "medium")]), False)["causes"] == [H_POWER]   # only the top level
assert decide_final(E([(H_SENSOR, "low"), (H_NO_FAULT, "medium")]), True)["kind"] == "no_fault"
assert decide_final(E([(H_SENSOR, "low"), (H_UNEXPLAINED, "medium")]), False) == \
    {"kind": "unexplained", "causes": [H_UNEXPLAINED], "confidence": "medium"}
assert decide_final(E([(H_SENSOR, "low"), (H_UNEXPLAINED, "low")]), False)["kind"] == "unresolved"

# ---- closed catalog ----
assert match_catalog("Tool Wear approaching wear-out threshold") == H_TOOL_WEAR
assert match_catalog("Heat\u2011dissipation margin narrowing as air cools") == H_HEAT
assert match_catalog("Sensor calibration drift") == H_SENSOR and match_catalog("Unexplained: x") == H_UNEXPLAINED
assert match_catalog("Control system/software fault or safety interlock") is None
assert match_catalog("Mechanical jam or bearing seizure") is None
req = dict(required_hypotheses({"predicted_class": "Power Failure + Overstrain Failure"}, c))
assert set(req) == {H_POWER, H_OVERSTRAIN}
assert dict(required_hypotheses(anom, c)) == {}
assert H_HEAT in dict(required_hypotheses(anom, ch))
assert H_TOOL_WEAR in dict(required_hypotheses(anom, cw, True))
assert H_TOOL_WEAR not in dict(required_hypotheses(anom, cw, False))              # routine check: not forced
hyps, notes = finalize_hypotheses([{"cause": "PLC glitch", "rationale": "r"}, {"cause": "Tool wear", "rationale": "x"}], anom, c, True, False)
assert [h["cause"] for h in hyps] == [H_TOOL_WEAR, H_NO_FAULT] and any("dropped" in n for n in notes)
hyps, _ = finalize_hypotheses([], anom, ch, False)
assert [h["cause"] for h in hyps] == [H_HEAT, H_UNEXPLAINED]

# ---- number grounding and duplicate facts ----
evid = " ".join(limit_check_facts(c)) + str([latest, older]) + str(anom) + DOC_CONSTANTS_TEXT
assert ungrounded_numbers("Temperature difference ~11.3 K above 8.6 K threshold", evid) == ["11.3"]
assert ungrounded_numbers("gap 9.7 K, speed 1589 rpm, tool wear 174, confidence 0.9998", evid) == []
facts, n = annotate_ungrounded(["gap 11.3 K", "speed 1589 rpm"], evid)
assert n == 1 and "UNVERIFIED" in facts[0] and "UNVERIFIED" not in facts[1]
assert drop_redundant_facts(["Speed 1589 rpm above 1380 rpm", "Air temperature latest 298.6 K", "Sensor statistics look normal",
                             "Gap 11.3 K"], " ".join(limit_check_facts(c))) == \
    ["Air temperature latest 298.6 K", "Sensor statistics look normal", "Gap 11.3 K"]
# false alarms seen in the first dev run: thin-space thousands and a manual constant
assert ungrounded_numbers("Latest tool_wear\u202f\u00d7\u202ftorque = 12\u202f337", "wear x torque 12337") == []
assert ungrounded_numbers("Technical documentation recommends replacement at 180\u2011200\u202fmin", DOC_CONSTANTS_TEXT) == []
assert ungrounded_numbers("a value of 12 999", "wear x torque 12337") == ["999"]      # plain spaces are NOT merged
# the window counts no longer reach the model: only the 'not a time series' note does
joined = " ".join(limit_check_facts(c))
assert "Across the" not in joined and "independent samples" in joined
print("\n".join(limit_check_facts(c)))
print("\nALL checks.py TESTS PASSED")
