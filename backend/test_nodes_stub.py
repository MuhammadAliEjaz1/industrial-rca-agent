"""Control-flow test of the NEW nodes.py with every external dependency stubbed (no Groq, no DB)."""
import sys, types
from types import SimpleNamespace as NS

class _Tool:
    def __init__(self, fn): self.fn = fn
    def invoke(self, args): return self.fn(args)

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

class Dumpable(NS):
    def model_dump(self): return dict(self.__dict__)

LLM_SCRIPT = {"conf": {}, "hyps": [], "root": None}

class FakeLLM:
    def with_structured_output(self, schema):
        name = schema if isinstance(schema, str) else schema.__name__
        def invoke(prompt):
            if LLM_SCRIPT.get("fail_next", 0) > 0:
                LLM_SCRIPT["fail_next"] -= 1
                raise Exception("Error code: 400 - tool_use_failed: Tool choice is required, but model did not call a tool")
            if name == "HypothesisList":
                return NS(hypotheses=[Dumpable(**h) for h in LLM_SCRIPT["hyps"]])
            if name == "HypothesisEvaluation":
                cause = next(h["cause"] for h in LLM_SCRIPT["hyps"] if h["cause"] in prompt)
                return Dumpable(cause=cause, supporting_evidence=["s"], contradicting_evidence=["c"],
                                reasoning="r", confidence=LLM_SCRIPT["conf"][cause])
            return Dumpable(**LLM_SCRIPT["root"])
        return NS(invoke=invoke)

mod("dotenv", load_dotenv=lambda: None)
mod("langchain_groq", ChatGroq=lambda **k: FakeLLM())
mod("graph.state", HypothesisList="HypothesisList", HypothesisEvaluation="HypothesisEvaluation",
    RootCauseDetermination="RootCauseDetermination")
HEALTHY = [dict(air_temperature=298.6, process_temperature=308.3, rotational_speed=1589, torque=38.0, tool_wear=174),
           dict(air_temperature=299.9, process_temperature=309.9, rotational_speed=1500, torque=40.0, tool_wear=100)]
ANOM = {"found": True, "predicted_class": "No Failure", "confidence": 0.999775, "reliability_warning": False,
        "reliability_note": None, "close_second_class": "Power Failure", "close_second_margin": 0.9995}
mod("tools.investigation_tools",
    get_sensor_data=_Tool(lambda a: {"found": True, "readings": HEALTHY, "reading_count": 2}),
    get_maintenance_history=_Tool(lambda a: {"records": [], "record_count": 0}),
    get_failure_history=_Tool(lambda a: {"incidents": [], "incident_count": 0}),
    run_anomaly_detection=_Tool(lambda a: ANOM),
    search_technical_documents=_Tool(lambda a: {"results": []}),
    calculate_sensor_statistics=_Tool(lambda a: {"channels": {"air_temperature": {"trend": "decreasing", "latest": 298.6}}}),
    search_similar_incidents=_Tool(lambda a: {"results": []}))
sys.path.insert(0, ".")
import graph.nodes as N

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

HYPS = [dict(cause="Sensor drift", rationale="r"), dict(cause="Power sag", rationale="r"),
        dict(cause="No fault: normal variation", rationale="r")]
ROOT = dict(most_supported_cause="Sensor drift", confidence_level="high", justification="j",
            observed_facts=["Temperature difference between process and air about 11.3 K", "Speed 1589 rpm"],
            ml_predictions=["Power Failure second class, similar confidence"], unsupported_speculation=["x"])

# Case 1: the exact failure from run 1 - LLM says HIGH for sensor drift on a healthy machine, twice
LLM_SCRIPT.update(hyps=HYPS, conf={"Sensor drift": "high", "Power sag": "medium", "No fault: normal variation": "low"}, root=dict(ROOT))
s = run_pipeline({"machine_id": "M001", "incident_description": "Routine automated check flagged this machine.", "failure_reported": False})
r = s["final_report"]
print("CASE 1 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"])
assert r["mode"] == "health_check"
assert all(e["confidence"] == "low" for e in r["candidate_root_causes"] if not e["cause"].startswith("No fault"))
assert r["most_supported_cause"].startswith("No fault identified")
assert not any("gather_more_evidence" == l["step"] for l in r["investigation_log"]), "all-low must not retry"
assert any("UNVERIFIED" in f and "11.3" in f for f in r["observed_facts"])
assert not any("Power Failure second class" in m for m in r["ml_predictions"]) and "No close runner-up" not in str(r["ml_predictions"])
assert any("[capped from" in e["reasoning"] for e in r["candidate_root_causes"])
assert "CONFIDENCE CAPS" in " ".join(l["summary"] for l in r["investigation_log"])

# Case 2: a real failure (heat dissipation limit violated, incident described): HIGH is allowed to stand
HOT = [dict(air_temperature=300.0, process_temperature=308.0, rotational_speed=1300, torque=40.0, tool_wear=50)]
N.get_sensor_data = _Tool(lambda a: {"found": True, "readings": HOT, "reading_count": 1})
ANOM.update(predicted_class="Heat Dissipation Failure", confidence=0.97, close_second_class=None, close_second_margin=None)
LLM_SCRIPT.update(hyps=[dict(cause="Cooling failure", rationale="r"), dict(cause="Unexplained: other", rationale="r")],
                  conf={"Cooling failure": "high", "Unexplained: other": "low"},
                  root=dict(ROOT, most_supported_cause="Cooling failure", confidence_level="high", observed_facts=["Speed 1300 rpm"]))
s = run_pipeline({"machine_id": "M002", "incident_description": "machine stopped"})
r = s["final_report"]
print("CASE 2 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"])
assert r["mode"] == "failure_investigation" and r["confidence_level"] == "high"

# Case 3: failure mode, nothing violated, best hypothesis medium -> retries up to MAX, never high
ANOM.update(predicted_class="Tool Wear Failure", confidence=0.6, reliability_warning=True, reliability_note="weak class")
N.get_sensor_data = _Tool(lambda a: {"found": True, "readings": HEALTHY, "reading_count": 2})
LLM_SCRIPT.update(hyps=[dict(cause="Tool wear", rationale="r"), dict(cause="Unexplained: other", rationale="r")],
                  conf={"Tool wear": "high", "Unexplained: other": "low"},
                  root=dict(ROOT, most_supported_cause="Tool wear", confidence_level="high", observed_facts=[]))
s = run_pipeline({"machine_id": "M003", "incident_description": "stopped"})
r = s["final_report"]
print("CASE 3 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"], "| retries:", s.get("retry_count", 0))
assert r["confidence_level"] == "medium" and s.get("retry_count", 0) == 2   # medium keeps retrying until MAX_RETRIES
assert "reliability_warning: YES" in r["ml_predictions"][0]

# Case 4: SAME healthy sensors but a failure WAS reported (e.g. a random failure): never 'No fault', and a
# 'medium' on the null hypothesis must not trigger retries
ANOM.update(predicted_class="No Failure", confidence=0.9998, reliability_warning=False, reliability_note=None,
            close_second_class="Power Failure", close_second_margin=0.9997)
LLM_SCRIPT.update(hyps=[dict(cause="Sensor drift", rationale="r"), dict(cause="Unexplained: random failure", rationale="r")],
                  conf={"Sensor drift": "low", "Unexplained: random failure": "medium"},
                  root=dict(ROOT, most_supported_cause="Unexplained: random failure", confidence_level="medium", observed_facts=[]))
s = run_pipeline({"machine_id": "M004", "incident_description": "machine stopped", "failure_reported": True})
r = s["final_report"]
print("CASE 4 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"], "| retries:", s.get("retry_count", 0))
assert r["mode"] == "failure_investigation" and not r["most_supported_cause"].startswith("No fault")
assert r["most_supported_cause"] == "Unexplained: random failure" and r["confidence_level"] == "medium"
assert s.get("retry_count", 0) == 0

# Case 5: failure_reported missing entirely -> defaults to failure investigation (safe default)
s = run_pipeline({"machine_id": "M005", "incident_description": "x"})
assert s["final_report"]["mode"] == "failure_investigation"

# Case 6: no specific fault AND the null hypothesis is low too -> Unresolved
LLM_SCRIPT.update(conf={"Sensor drift": "low", "Unexplained: random failure": "low"})
s = run_pipeline({"machine_id": "M006", "incident_description": "machine stopped", "failure_reported": True})
assert s["final_report"]["most_supported_cause"] == "Unresolved - insufficient evidence"

# Case 7: Groq 'tool_use_failed' twice -> retried and still completes; three times -> raises
LLM_SCRIPT.update(fail_next=2)
s = run_pipeline({"machine_id": "M007", "incident_description": "x", "failure_reported": False})
assert s["final_report"]["mode"] == "health_check" and LLM_SCRIPT["fail_next"] == 0
LLM_SCRIPT.update(fail_next=3)
try:
    run_pipeline({"machine_id": "M008", "incident_description": "x", "failure_reported": False})
    raise SystemExit("expected an error after 3 failures")
except Exception as e:
    assert "tool_use_failed" in str(e), e
LLM_SCRIPT.update(fail_next=0)
# Case 8: routine check, ML says No Failure, but tool wear 215 -> NOT a health check
WORN = [dict(air_temperature=298.0, process_temperature=308.3, rotational_speed=1500, torque=30.0, tool_wear=215)]
N.get_sensor_data = _Tool(lambda a: {"found": True, "readings": WORN, "reading_count": 1})
ANOM.update(predicted_class="No Failure", confidence=0.9998, close_second_class=None, close_second_margin=None)
LLM_SCRIPT.update(hyps=[dict(cause="Tool wear", rationale="r"), dict(cause="No fault: normal", rationale="r")],
                  conf={"Tool wear": "high", "No fault: normal": "low"},
                  root=dict(ROOT, most_supported_cause="Tool wear", confidence_level="high", observed_facts=[]))
s = run_pipeline({"machine_id": "M009", "incident_description": "x", "failure_reported": False})
r = s["final_report"]
print("CASE 8 ->", r["mode"], "|", r["most_supported_cause"], "|", r["confidence_level"])
assert r["mode"] == "failure_investigation" and not r["most_supported_cause"].startswith("No fault")
print("\nALL STUB CONTROL-FLOW TESTS PASSED (8 cases)")
