"""Control-flow test of nodes.py with every external dependency stubbed (no Groq, no DB). From backend/: python test_nodes_stub.py"""
import re, sys, types
from types import SimpleNamespace as NS

class _Tool:
    def __init__(self, fn): self.fn = fn
    def invoke(self, args): return self.fn(args)

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

class Dumpable(NS):
    def model_dump(self): return dict(self.__dict__)

LLM = {"hyps": [], "conf": {}, "root": None, "fail_next": 0, "calls": 0}

class FakeLLM:
    def with_structured_output(self, schema):
        name = schema if isinstance(schema, str) else schema.__name__
        def invoke(prompt):
            if LLM["fail_next"] > 0:
                LLM["fail_next"] -= 1
                raise Exception("Error code: 400 - tool_use_failed: Tool choice is required, but model did not call a tool")
            LLM["calls"] += 1
            if name == "HypothesisList":
                return NS(hypotheses=[Dumpable(**h) for h in LLM["hyps"]])
            if name == "HypothesisEvaluation":
                cause = re.search(r"CANDIDATE CAUSE: (.*)", prompt).group(1).strip()
                section = prompt.lower().split("sensor statistics")[1].split("technical documentation")[0]
                assert "trend=" not in section and "decreasing" not in section, "trend values must not reach the evaluator"
                return Dumpable(cause=cause, supporting_evidence=["s"], contradicting_evidence=["c"],
                                reasoning="r", confidence=LLM["conf"].get(cause, "low"))
            assert "OUTCOME DECIDED BY THE SYSTEM" in prompt       # the LLM is told the outcome, it does not choose it
            return Dumpable(**LLM["root"])
        return NS(invoke=invoke)

mod("dotenv", load_dotenv=lambda: None)
mod("langchain_groq", ChatGroq=lambda **k: FakeLLM())
mod("graph.state", HypothesisList="HypothesisList", HypothesisEvaluation="HypothesisEvaluation",
    RootCauseDetermination="RootCauseDetermination")

R = lambda a, p, rpm, t, w: dict(air_temperature=a, process_temperature=p, rotational_speed=rpm, torque=t, tool_wear=w)
HEALTHY = [R(298.6, 308.3, 1589, 38.0, 174), R(299.9, 309.9, 1500, 40.0, 100)]
HOT = [R(300.0, 308.0, 1300, 40.0, 50)]
WORN = [R(298.0, 308.3, 1500, 30.0, 215)]
POWER_LOW = [R(298.9, 309.1, 2861, 4.6, 143)]                 # real incident 1
POWER_OVER = [R(298.9, 309.0, 1410, 65.7, 191)]               # real incident 2: power 9701 W, wear x torque 12549
POWER_LOW_WORN = [R(298.9, 309.1, 2861, 4.6, 215)]            # power out of range AND tool wear in the failure range
NOFAIL = {"found": True, "predicted_class": "No Failure", "confidence": 0.999775, "reliability_warning": False,
          "reliability_note": None, "close_second_class": "Power Failure", "close_second_margin": 0.9995}
STATE = {"readings": HEALTHY, "anomaly": dict(NOFAIL)}

CALLS = []
def rec(name, fn):
    def wrapped(a):
        CALLS.append((name, dict(a)))
        return fn(a)
    return _Tool(wrapped)

mod("tools.investigation_tools",
    get_sensor_data=rec("get_sensor_data", lambda a: {"found": True, "readings": STATE["readings"], "reading_count": len(STATE["readings"])}),
    get_maintenance_history=rec("get_maintenance_history", lambda a: {"records": [], "record_count": 0}),
    get_failure_history=rec("get_failure_history", lambda a: {"incidents": [], "incident_count": 0}),
    run_anomaly_detection=rec("run_anomaly_detection", lambda a: STATE["anomaly"]),
    search_technical_documents=_Tool(lambda a: {"results": []}),
    calculate_sensor_statistics=rec("calculate_sensor_statistics", lambda a: {"channels": {
        "air_temperature": {"trend": "decreasing", "latest": 298.6, "mean": 299.9},
        "process_temperature": {"trend": "stable", "latest": 308.3, "mean": 309.75}}}),
    search_similar_incidents=_Tool(lambda a: {"results": []}))
sys.path.insert(0, ".")
import graph.nodes as N
from graph.checks import H_TOOL_WEAR, H_HEAT, H_POWER, H_OVERSTRAIN, H_SENSOR, H_NO_FAULT, H_UNEXPLAINED

def run_pipeline(state):
    state = dict(state, investigation_log=[])
    def upd(d):
        for k, v in d.items():
            state[k] = state.get(k, []) + v if k == "investigation_log" else v
    for node in (N.gather_context, N.run_anomaly_model, N.gather_evidence, N.generate_hypotheses, N.investigate_hypotheses):
        upd(node(state))
    while N.route_after_investigation(state) == "insufficient":
        upd(N.gather_more_evidence(state)); upd(N.investigate_hypotheses(state))
    upd(N.determine_root_cause(state)); upd(N.generate_report(state))
    return state

def setup(readings, anomaly=None, hyps=(), conf=None, root=None):
    STATE["readings"] = readings
    STATE["anomaly"] = dict(NOFAIL, **(anomaly or {}))
    base = dict(most_supported_cause="whatever the LLM says is ignored", confidence_level="high", justification="j",
                observed_facts=[], ml_predictions=["Power Failure second class, similar confidence"],
                unsupported_speculation=["x"])
    base.update(root or {})
    LLM.update(hyps=[dict(cause=c, rationale="r") for c in hyps], conf=conf or {}, root=base, calls=0)

def causes(r): return {e["cause"]: e["confidence"] for e in r["candidate_root_causes"]}
def log(s): return " ".join(l["summary"] for l in s["investigation_log"])
ROUTINE = {"incident_description": "Routine automated check flagged this machine.", "failure_reported": False}
FAILURE = {"incident_description": "A failure was reported.", "failure_reported": True}

# C1: the run-1 failure: LLM says HIGH for sensor drift on a healthy machine; also proposes an off-catalog cause
setup(HEALTHY, hyps=[H_SENSOR, H_TOOL_WEAR, "Gremlins in the PLC"], conf={H_SENSOR: "high", H_TOOL_WEAR: "medium"},
      root=dict(observed_facts=["Temperature difference between process and air about 11.3 K", "Speed 1589 rpm",
                                "Mean temperature gap over history about 9.85 K"]))
s = run_pipeline(dict(machine_id="M001", **ROUTINE)); r = s["final_report"]; c = causes(r)
print("C1 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"], "|", c)
assert r["mode"] == "health_check" and r["most_supported_cause"].startswith("No fault identified") and not r["advisories"]
assert c == {H_SENSOR: "low", H_TOOL_WEAR: "low", H_NO_FAULT: "medium"}
assert "Gremlins" in log(s) and "dropped" in log(s) and "settled by rule without the model" in log(s)
assert any("Not sent to the model" in e["reasoning"] for e in r["candidate_root_causes"])
assert all("llm_confidence" not in e for e in r["candidate_root_causes"])      # nothing was sent to the model for these
assert LLM["calls"] == 2, LLM["calls"]                                           # a whole health check = hypotheses + explanation
assert any("11.3" in f and "UNVERIFIED" in f for f in r["observed_facts"])           # invented number flagged
assert not any("9.85" in f and "UNVERIFIED" in f for f in r["observed_facts"])       # the derived mean gap (9.85) is grounded
assert not any(f == "Speed 1589 rpm" for f in r["observed_facts"])
assert [x.split(" (")[0] for x in r["unsupported_speculation"]] == [H_SENSOR, H_TOOL_WEAR]
assert "gather_more_evidence" not in [l["step"] for l in s["investigation_log"]]

# C2: the pilot's inc-117: heat limit really met, classifier agrees, but the LLM rated heat LOW -> the rule wins
setup(HOT, anomaly=dict(predicted_class="Heat Dissipation Failure", confidence=0.97, close_second_class=None, close_second_margin=None),
      hyps=[H_SENSOR, H_TOOL_WEAR], conf={H_HEAT: "low", H_TOOL_WEAR: "high", H_SENSOR: "medium"})
s = run_pipeline(dict(machine_id="M002", **FAILURE)); r = s["final_report"]; c = causes(r)
print("C2 ->", r["most_supported_cause"], "|", r["confidence_level"], "|", c)
assert c[H_HEAT] == "high" and c[H_TOOL_WEAR] == "low" and c[H_SENSOR] == "low"
assert r["most_supported_cause"] == H_HEAT and r["confidence_level"] == "high"
assert any(e.get("llm_confidence") == "low" and e["confidence"] == "high" for e in r["candidate_root_causes"])
assert "added 'Heat dissipation failure'" in log(s)
assert LLM["calls"] == 4, LLM["calls"]       # hypotheses + heat + 'unexplained' + explanation; sensor and tool wear settled by rule

# C3: ML predicts Tool Wear (reliability warning), sensors look fine: medium at most, never retried (MAX_RETRIES=0)
setup(HEALTHY, anomaly=dict(predicted_class="Tool Wear Failure", confidence=0.6, reliability_warning=True,
                            reliability_note="weak class", close_second_class=None, close_second_margin=None),
      hyps=[H_SENSOR], conf={H_TOOL_WEAR: "high"})
s = run_pipeline(dict(machine_id="M003", **FAILURE)); r = s["final_report"]
print("C3 ->", r["most_supported_cause"], "|", r["confidence_level"], "| retries:", s.get("retry_count", 0))
assert r["most_supported_cause"] == H_TOOL_WEAR and r["confidence_level"] == "medium" and s.get("retry_count", 0) == 0
assert "reliability_warning: YES" in r["ml_predictions"][0]

# C4: failure reported, healthy sensors, 'Unexplained' rated medium -> that is the answer
setup(HEALTHY, hyps=[H_SENSOR], conf={H_UNEXPLAINED: "medium"})
s = run_pipeline(dict(machine_id="M004", **FAILURE)); r = s["final_report"]
print("C4 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"])
assert r["mode"] == "failure_investigation" and r["most_supported_cause"] == H_UNEXPLAINED and r["confidence_level"] == "medium"

# C5: failure_reported missing -> safe default is a failure investigation
setup(HEALTHY, hyps=[H_SENSOR])
assert run_pipeline({"machine_id": "M005", "incident_description": "x"})["final_report"]["mode"] == "failure_investigation"

# C6: everything low, including 'Unexplained' -> Unresolved
setup(HEALTHY, hyps=[H_SENSOR])
s = run_pipeline(dict(machine_id="M006", **FAILURE))
assert s["final_report"]["most_supported_cause"] == "Unresolved - insufficient evidence"

# C7: Groq 'tool_use_failed' twice -> retried and completes; three times -> raises
setup(HEALTHY, hyps=[H_SENSOR]); LLM["fail_next"] = 2
assert run_pipeline(dict(machine_id="M007", **ROUTINE))["final_report"]["mode"] == "health_check" and LLM["fail_next"] == 0
LLM["fail_next"] = 3
try:
    run_pipeline(dict(machine_id="M008", **ROUTINE)); raise SystemExit("expected an error after 3 failures")
except Exception as e:
    assert "tool_use_failed" in str(e), e
LLM["fail_next"] = 0

# C8: ROUTINE check, ML says No Failure, tool wear 215: still a health check, with an ADVISORY (not a diagnosis)
setup(WORN, hyps=[H_SENSOR, H_TOOL_WEAR], conf={H_TOOL_WEAR: "high"})
s = run_pipeline(dict(machine_id="M009", **ROUTINE)); r = s["final_report"]
print("C8 ->", r["mode"], "|", r["most_supported_cause"], "|", r["advisories"][:1])
assert r["mode"] == "health_check" and r["most_supported_cause"].startswith("No fault identified")
assert r["advisories"] and "ADVISORY" in r["justification"] and causes(r)[H_TOOL_WEAR] == "low"

# C8b: same reading but a failure WAS reported: tool wear is forced in and reaches 'medium'
setup(WORN, hyps=[H_SENSOR], conf={H_TOOL_WEAR: "high"})
s = run_pipeline(dict(machine_id="M009", **FAILURE)); r = s["final_report"]
print("C8b ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"])
assert r["most_supported_cause"] == H_TOOL_WEAR and r["confidence_level"] == "medium" and not r["advisories"]

# C9: as_of travels from the graph input to EVERY time-aware tool
setup(HEALTHY, hyps=[H_SENSOR]); CALLS.clear()
run_pipeline(dict(machine_id="M010", **FAILURE, as_of="2025-03-03 08:00:00.000001", incident_id=5))
seen = {name: args.get("as_of") for name, args in CALLS}
assert set(seen) == {"get_sensor_data", "calculate_sensor_statistics", "run_anomaly_detection",
                     "get_maintenance_history", "get_failure_history"} and all(v == "2025-03-03 08:00:00.000001" for v in seen.values())
CALLS.clear(); run_pipeline(dict(machine_id="M011", **FAILURE))
assert all(args.get("as_of") is None for _, args in CALLS)

# C10: compound failure (real incident 2): both causes rated high, named together; tool wear (191 min) stays low
setup(POWER_OVER, anomaly=dict(predicted_class="Power Failure + Overstrain Failure", confidence=0.99, close_second_class=None,
                               close_second_margin=None), hyps=[H_POWER], conf={})
s = run_pipeline(dict(machine_id="M012", **FAILURE)); r = s["final_report"]; c = causes(r)
print("C10 ->", r["most_supported_cause"], "|", r["confidence_level"], "|", c)
assert r["most_supported_cause"] == f"{H_POWER} + {H_OVERSTRAIN}" and r["confidence_level"] == "high"
assert c.get(H_TOOL_WEAR, "low") == "low"

# C11: a hard limit is violated AND tool wear is 215: the hard limit explains it, tool wear is demoted to low
setup(POWER_LOW_WORN, anomaly=dict(predicted_class="Power Failure", confidence=0.99, close_second_class=None, close_second_margin=None),
      hyps=[H_POWER], conf={H_TOOL_WEAR: "high"})
s = run_pipeline(dict(machine_id="M013", **FAILURE)); r = s["final_report"]; c = causes(r)
print("C11 ->", r["most_supported_cause"], "|", c)
assert r["most_supported_cause"] == H_POWER and c[H_TOOL_WEAR] == "low" and c[H_POWER] == "high"

# C12: an OLDER reading that broke a limit does not turn a routine check on a healthy latest reading into an investigation
setup([HEALTHY[0], HOT[0]], hyps=[H_SENSOR])
assert run_pipeline(dict(machine_id="M014", **ROUTINE))["final_report"]["mode"] == "health_check"

# C13: the retry loop still works when it is switched on
N.MAX_RETRIES = 2
setup(WORN, hyps=[H_SENSOR], conf={})
s = run_pipeline(dict(machine_id="M015", **FAILURE))
assert s.get("retry_count", 0) == 2, s.get("retry_count")          # tool wear is 'medium' -> promising -> loops to the limit
N.MAX_RETRIES = 0

# _shorten: never stops mid-word, marks the cut
long = "All computed limit checks are within documented normal ranges (heat dissipation gap 9.7 K > 8.6 K, speed 1589 rpm, power 5674 W within range, tool wear 174 min, tool wear x torque 5933 below limit)"
short = N._shorten(long)
assert short.endswith(" ...") and len(short) <= 165 and long.startswith(short[:-4]) and not short[:-4].endswith(" ")
assert N._shorten("short text") == "short text"
print("\nALL STUB CONTROL-FLOW TESTS PASSED (13 cases + shorten)")
