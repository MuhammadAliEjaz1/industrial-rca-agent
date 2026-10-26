"""Control-flow test of nodes.py with every external dependency stubbed (no Groq, no DB)."""
import re, sys, types
from types import SimpleNamespace as NS

class _Tool:
    def __init__(self, fn): self.fn = fn
    def invoke(self, args): return self.fn(args)

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

class Dumpable(NS):
    def model_dump(self): return dict(self.__dict__)

LLM = {"hyps": [], "conf": {}, "root": None, "fail_next": 0}

class FakeLLM:
    def with_structured_output(self, schema):
        name = schema if isinstance(schema, str) else schema.__name__
        def invoke(prompt):
            if LLM["fail_next"] > 0:
                LLM["fail_next"] -= 1
                raise Exception("Error code: 400 - tool_use_failed: Tool choice is required, but model did not call a tool")
            if name == "HypothesisList":
                return NS(hypotheses=[Dumpable(**h) for h in LLM["hyps"]])
            if name == "HypothesisEvaluation":
                cause = re.search(r"CANDIDATE CAUSE: (.*)", prompt).group(1).strip()
                assert "trend" not in prompt.lower().split("sensor statistics")[1].split("technical documentation")[0].replace(
                    "no trend information", ""), "trend labels must not reach the evaluator"
                return Dumpable(cause=cause, supporting_evidence=["s"], contradicting_evidence=["c"],
                                reasoning="r", confidence=LLM["conf"].get(cause, "low"))
            return Dumpable(**LLM["root"])
        return NS(invoke=invoke)

mod("dotenv", load_dotenv=lambda: None)
mod("langchain_groq", ChatGroq=lambda **k: FakeLLM())
mod("graph.state", HypothesisList="HypothesisList", HypothesisEvaluation="HypothesisEvaluation",
    RootCauseDetermination="RootCauseDetermination")

HEALTHY = [dict(air_temperature=298.6, process_temperature=308.3, rotational_speed=1589, torque=38.0, tool_wear=174),
           dict(air_temperature=299.9, process_temperature=309.9, rotational_speed=1500, torque=40.0, tool_wear=100)]
HOT = [dict(air_temperature=300.0, process_temperature=308.0, rotational_speed=1300, torque=40.0, tool_wear=50)]
WORN = [dict(air_temperature=298.0, process_temperature=308.3, rotational_speed=1500, torque=30.0, tool_wear=215)]
NOFAIL = {"found": True, "predicted_class": "No Failure", "confidence": 0.999775, "reliability_warning": False,
          "reliability_note": None, "close_second_class": "Power Failure", "close_second_margin": 0.9995}
STATE = {"readings": HEALTHY, "anomaly": dict(NOFAIL)}

mod("tools.investigation_tools",
    get_sensor_data=_Tool(lambda a: {"found": True, "readings": STATE["readings"], "reading_count": len(STATE["readings"])}),
    get_maintenance_history=_Tool(lambda a: {"records": [], "record_count": 0}),
    get_failure_history=_Tool(lambda a: {"incidents": [], "incident_count": 0}),
    run_anomaly_detection=_Tool(lambda a: STATE["anomaly"]),
    search_technical_documents=_Tool(lambda a: {"results": []}),
    calculate_sensor_statistics=_Tool(lambda a: {"channels": {"air_temperature": {"trend": "decreasing", "latest": 298.6, "mean": 299.9}}}),
    search_similar_incidents=_Tool(lambda a: {"results": []}))
sys.path.insert(0, ".")
import graph.nodes as N
from graph.checks import H_TOOL_WEAR, H_HEAT, H_SENSOR, H_NO_FAULT, H_UNEXPLAINED

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
    base = dict(most_supported_cause="x", confidence_level="high", justification="j", observed_facts=[],
                ml_predictions=["Power Failure second class, similar confidence"], unsupported_speculation=["x"])
    base.update(root or {})
    LLM.update(hyps=[dict(cause=c, rationale="r") for c in hyps], conf=conf or {}, root=base)

def causes(r): return {e["cause"]: e["confidence"] for e in r["candidate_root_causes"]}
def log(s): return " ".join(l["summary"] for l in s["investigation_log"])

# C1: the run-1 failure: LLM says HIGH for sensor drift on a healthy machine; also proposes an off-catalog cause
setup(HEALTHY, hyps=[H_SENSOR, H_TOOL_WEAR, "Gremlins in the PLC"],
      conf={H_SENSOR: "high", H_TOOL_WEAR: "medium"},
      root=dict(observed_facts=["Temperature difference between process and air about 11.3 K", "Speed 1589 rpm"]))
s = run_pipeline({"machine_id": "M001", "incident_description": "Routine check flagged this machine.", "failure_reported": False})
r = s["final_report"]; c = causes(r)
print("C1 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"], "|", c)
assert r["mode"] == "health_check" and r["most_supported_cause"].startswith("No fault identified")
assert c == {H_SENSOR: "low", H_TOOL_WEAR: "low", H_NO_FAULT: "medium"}          # caps + deterministic null, no Gremlins
assert "Gremlins" in log(s) and "dropped" in log(s) and "gather_more_evidence" not in [l["step"] for l in s["investigation_log"]]
assert any("11.3" in f and "UNVERIFIED" in f for f in r["observed_facts"])         # invented number flagged
assert not any(f == "Speed 1589 rpm" for f in r["observed_facts"])                  # duplicate of a computed fact dropped
assert "CONFIDENCE CAPS" in log(s) and any("[capped from" in e["reasoning"] for e in r["candidate_root_causes"])

# C2: heat-dissipation limit really violated; the LLM forgets to propose it -> forced in, and HIGH is allowed
setup(HOT, anomaly=dict(predicted_class="Heat Dissipation Failure", confidence=0.97, close_second_class=None,
                        close_second_margin=None),
      hyps=[H_SENSOR], conf={H_HEAT: "high", H_SENSOR: "medium"}, root=dict(most_supported_cause=H_HEAT))
s = run_pipeline({"machine_id": "M002", "incident_description": "machine stopped", "failure_reported": True})
r = s["final_report"]; c = causes(r)
print("C2 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"], "|", c)
assert r["mode"] == "failure_investigation" and c[H_HEAT] == "high" and r["confidence_level"] == "high"
assert "added 'Heat dissipation failure'" in log(s)

# C3: ML predicts Tool Wear with a reliability warning, nothing violated: never above medium, retries until MAX
setup(HEALTHY, anomaly=dict(predicted_class="Tool Wear Failure", confidence=0.6, reliability_warning=True,
                            reliability_note="weak class", close_second_class=None, close_second_margin=None),
      hyps=[H_SENSOR], conf={H_TOOL_WEAR: "high"}, root=dict(most_supported_cause=H_TOOL_WEAR))
s = run_pipeline({"machine_id": "M003", "incident_description": "stopped", "failure_reported": True})
r = s["final_report"]
print("C3 ->", r["most_supported_cause"], "|", r["confidence_level"], "| retries:", s.get("retry_count", 0))
assert causes(r)[H_TOOL_WEAR] == "medium" and r["confidence_level"] == "medium" and s.get("retry_count", 0) == 2
assert "reliability_warning: YES" in r["ml_predictions"][0]

# C4: failure reported, healthy sensors, the 'Unexplained' reading is medium -> chosen, no retries
setup(HEALTHY, hyps=[H_SENSOR], conf={H_UNEXPLAINED: "medium"}, root=dict(most_supported_cause=H_SENSOR))
s = run_pipeline({"machine_id": "M004", "incident_description": "machine stopped", "failure_reported": True})
r = s["final_report"]
print("C4 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"], "| retries:", s.get("retry_count", 0))
assert r["mode"] == "failure_investigation" and r["most_supported_cause"] == H_UNEXPLAINED
assert r["confidence_level"] == "medium" and s.get("retry_count", 0) == 0

# C5: failure_reported missing -> safe default is a failure investigation
setup(HEALTHY, hyps=[H_SENSOR])
assert run_pipeline({"machine_id": "M005", "incident_description": "x"})["final_report"]["mode"] == "failure_investigation"

# C6: everything low, including 'Unexplained' -> Unresolved
setup(HEALTHY, hyps=[H_SENSOR], conf={})
s = run_pipeline({"machine_id": "M006", "incident_description": "machine stopped", "failure_reported": True})
assert s["final_report"]["most_supported_cause"] == "Unresolved - insufficient evidence"

# C7: Groq 'tool_use_failed' twice -> retried and completes; three times -> raises
setup(HEALTHY, hyps=[H_SENSOR]); LLM["fail_next"] = 2
s = run_pipeline({"machine_id": "M007", "incident_description": "x", "failure_reported": False})
assert s["final_report"]["mode"] == "health_check" and LLM["fail_next"] == 0
LLM["fail_next"] = 3
try:
    run_pipeline({"machine_id": "M008", "incident_description": "x", "failure_reported": False})
    raise SystemExit("expected an error after 3 failures")
except Exception as e:
    assert "tool_use_failed" in str(e), e
LLM["fail_next"] = 0

# C8: routine check, ML says No Failure, but tool wear 215 -> investigation; tool wear forced in, capped at medium
setup(WORN, hyps=[H_SENSOR], conf={H_TOOL_WEAR: "high"}, root=dict(most_supported_cause=H_TOOL_WEAR))
s = run_pipeline({"machine_id": "M009", "incident_description": "x", "failure_reported": False})
r = s["final_report"]
print("C8 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"])
assert r["mode"] == "failure_investigation" and causes(r)[H_TOOL_WEAR] == "medium" and r["confidence_level"] == "medium"
assert "added 'Tool wear failure'" in log(s)

print("\nALL STUB CONTROL-FLOW TESTS PASSED (8 cases)")
