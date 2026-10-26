from graph.checks import *

# M001-like latest reading from the real run (torque is a made-up typical value: it was not in the output)
latest = dict(air_temperature=298.6, process_temperature=308.3, rotational_speed=1589, torque=38.0, tool_wear=174)
older = dict(air_temperature=299.9, process_temperature=309.9, rotational_speed=1500, torque=40.0, tool_wear=100)
c = compute_limit_checks([latest, older])
assert round(c["latest"]["temp_diff_k"], 1) == 9.7, c["latest"]
assert c["violated"] == [], c
assert not c["latest"]["heat_dissipation"]
anom = {"predicted_class": "No Failure", "confidence": 0.999775, "close_second_class": "Power Failure",
        "close_second_margin": 0.9995}
assert is_health_check(anom, False, c)
assert not is_health_check(anom, True, c)   # a reported failure is never a health check
assert not near_tie(anom)                              # margin 0.9995 is NOT a near tie
assert near_tie({"close_second_class": "X", "close_second_margin": 0.1})

# a real heat-dissipation case
hot = dict(air_temperature=300.0, process_temperature=308.0, rotational_speed=1300, torque=40.0, tool_wear=50)
ch = compute_limit_checks([hot])
assert ch["violated"] == ["heat_dissipation"], ch
assert not is_health_check({"predicted_class": "Heat Dissipation Failure"}, False, ch)

# caps
assert cap_confidence("Sensor drift", "high", c, anom)[0] == "low"          # healthy + no limit -> low
assert cap_confidence("Tool wear", "high", c, {"predicted_class": "Tool Wear Failure", "confidence": .6})[0] == "medium"
assert cap_confidence("Heat dissipation", "high", ch, {"predicted_class": "Heat Dissipation Failure", "confidence": .9})[0] == "high"
assert cap_confidence("No fault: normal variation", "high", c, anom) == ("high", None)
assert cap_confidence("Sensor drift", "low", c, anom) == ("low", None)

# number grounding: the real 11.3 K error must be caught, the real 9.7 K must not
ev = " ".join(limit_check_facts(c)) + str([latest, older]) + str(anom) + DOC_CONSTANTS_TEXT
assert ungrounded_numbers("Temperature difference ~11.3 K above 8.6 K threshold", ev) == ["11.3"]
assert ungrounded_numbers("gap 9.7 K, speed 1589 rpm, tool wear 174, confidence 0.9998", ev) == []
facts, n = annotate_ungrounded(["gap 11.3 K", "speed 1589 rpm"], ev)
assert n == 1 and "UNVERIFIED" in facts[0] and "UNVERIFIED" not in facts[1]
# tool wear in the documented failure range: not a limit violation, but never a 'health check'
worn = dict(air_temperature=298.0, process_temperature=308.3, rotational_speed=1500, torque=30.0, tool_wear=215)
cw = compute_limit_checks([worn])
assert cw["violated"] == [] and cw["latest"]["tool_wear_at_failure_range"]
assert not is_health_check(anom, False, cw)
assert cap_confidence("Tool wear", "high", cw, anom)[0] == "medium"      # not squeezed to low
assert cap_confidence("Tool wear", "high", c, anom)[0] == "low"         # healthy reading: still low
print("\n".join(limit_check_facts(c)))
print("\nALL checks.py TESTS PASSED")
