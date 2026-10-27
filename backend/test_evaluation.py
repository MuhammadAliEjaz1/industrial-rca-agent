"""Unit tests for the Phase 8 scoring helpers (no database, no model, no LLM). From backend/:  python test_evaluation.py"""
from evaluation.common import truth_components, ml_components, rule_components, score_case
from graph.checks import H_HEAT, H_POWER, H_OVERSTRAIN, H_TOOL_WEAR

# truth labels -> expected causes
assert truth_components("Power Failure") == {H_POWER}
assert truth_components("Power Failure + Overstrain Failure") == {H_POWER, H_OVERSTRAIN}
assert truth_components("Tool Wear Failure + Random Failure") == {H_TOOL_WEAR}     # random has no sensor signature
assert truth_components("Unknown") == set() and truth_components("No Failure") == set()
# classifier classes -> named causes
assert ml_components("Heat Dissipation Failure + Overstrain Failure") == {H_HEAT, H_OVERSTRAIN}
assert ml_components("Multiple Failure Causes (rare combination)") == set()        # names nothing specific
assert ml_components("No Failure") == set() and ml_components("Unknown") == set()
# scoring
assert score_case({H_POWER}, {H_POWER}) == "exact"
assert score_case(set(), set()) == "exact"                                         # healthy / Unknown: correctly names nothing
assert score_case({H_POWER, H_OVERSTRAIN}, {H_POWER}) == "partial"
assert score_case({H_POWER}, {H_POWER, H_TOOL_WEAR}) == "partial"
assert score_case({H_POWER}, {H_HEAT}) == "wrong"
assert score_case({H_POWER}, set()) == "missed"
assert score_case(set(), {H_HEAT}) == "overclaim"
# rules: the two real incidents you pasted earlier
inc1 = dict(air_temperature=298.9, process_temperature=309.1, rotational_speed=2861, torque=4.6, tool_wear=143)
inc2 = dict(air_temperature=298.9, process_temperature=309.0, rotational_speed=1410, torque=65.7, tool_wear=191)
assert rule_components(inc1) == {H_POWER} and rule_components(inc2) == {H_POWER, H_OVERSTRAIN}
worn = dict(air_temperature=298.0, process_temperature=308.3, rotational_speed=1500, torque=30.0, tool_wear=215)
assert rule_components(worn) == {H_TOOL_WEAR} and rule_components(worn, include_tool_wear_signal=False) == set()
print("ALL evaluation scoring TESTS PASSED")
