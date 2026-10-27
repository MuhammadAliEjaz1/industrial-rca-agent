"""Tests for the Phase 8 runner and report helpers (no database, no Groq). From backend/:  python test_runner.py"""
import importlib.util, json, tempfile
from pathlib import Path

from evaluation.report_lib import summarize, format_report, wilson
from graph.checks import H_POWER, H_HEAT, H_SENSOR, H_UNEXPLAINED

spec = importlib.util.spec_from_file_location("run_agent", Path("evaluation/08_run_agent.py"))
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)

inc = dict(case_id="inc-1", kind="incident", incident_id=1, machine_id="M025", as_of="2025-03-04 05:56:32.193807",
           truth_raw="Power Failure", expected=[H_POWER], in_test=True)
hea = dict(case_id="healthy-7", kind="healthy", incident_id=None, machine_id="M002", as_of="2025-03-05 01:00:00.000001",
           truth_raw="No Failure", expected=[], in_test=True)

# the agent's input never contains the answer
s = ra.case_state(inc)
assert s["failure_reported"] is True and s["incident_id"] == 1 and s["as_of"] == inc["as_of"]
assert "Power" not in s["incident_description"] and "Failure" not in s["incident_description"]
s = ra.case_state(hea)
assert s["failure_reported"] is False and s["incident_id"] is None and s["as_of"] == hea["as_of"]

# what the agent 'names': non-null candidates at medium or high
rep = {"candidate_root_causes": [{"cause": H_POWER, "confidence": "high"}, {"cause": H_SENSOR, "confidence": "low"},
                                 {"cause": H_UNEXPLAINED, "confidence": "high"}, {"cause": H_HEAT, "confidence": "medium"}]}
assert ra.agent_components(rep) == {H_POWER, H_HEAT}

# the pilot order spreads over labels
cases = [dict(inc, case_id=f"inc-{i}", truth_raw="Power Failure") for i in range(5)] + \
        [dict(inc, case_id=f"inc-{10 + i}", truth_raw="Tool Wear Failure") for i in range(5)] + [hea]
first3 = [c["truth_raw"] for c in ra.order_cases(cases)[:3]]
assert set(first3) == {"Power Failure", "Tool Wear Failure", "No Failure"}, first3

# run(): writes rows and reports, records errors, and resumes without repeating finished cases
class FakeGraph:
    def __init__(self): self.calls = []
    def invoke(self, state):
        self.calls.append(state["machine_id"])
        if state["machine_id"] == "BOOM":
            raise RuntimeError("Error code: 500 - server error")
        return {"retry_count": 1, "sensor_readings": [1, 2, 3], "sensor_statistics": {"reading_count": 9},
                "final_report": {"mode": "failure_investigation", "most_supported_cause": H_POWER, "confidence_level": "high",
                                 "candidate_root_causes": [{"cause": H_POWER, "confidence": "high", "confidence_note": "x"}],
                                 "observed_facts": ["a", "b [UNVERIFIED NUMBER(S): 11.3]"],
                                 "ml_analysis": {"predicted_class": "Power Failure"}}}
with tempfile.TemporaryDirectory() as d:
    out, reps = Path(d) / "r.jsonl", Path(d) / "reports"
    boom = dict(inc, case_id="inc-99", machine_id="BOOM")
    g = FakeGraph()
    st = ra.run([inc, boom], g, out, reps, pause=0)
    assert st["ok"] == 1 and st["error"] == 1 and (reps / "inc-1.json").exists()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows[0]["score"] == "exact" and rows[0]["unverified_facts"] == 1 and rows[0]["n_history_readings"] == 9
    assert rows[1]["status"] == "error" and "500" in rows[1]["error"]
    g2 = FakeGraph()
    ra.run([inc, boom], g2, out, reps, pause=0)
    assert g2.calls == ["BOOM"], g2.calls              # finished case skipped, failed case retried

# Groq's DAILY token limit stops the run at once, writes no error row, and loses nothing already finished
class TpdGraph:
    def invoke(self, state):
        raise RuntimeError("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
                           "Limit 200000, Used 198756, Requested 2263. Please try again in 7m20.208s.")
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "r.jsonl"
    st = ra.run([inc, hea], TpdGraph(), out, Path(d) / "rep", pause=0)
    assert st["stopped_daily_limit"] is True and (not out.exists() or out.read_text() == "")
assert ra.is_daily_limit("... on tokens per day (TPD): Limit 200000") and not ra.is_daily_limit("... on tokens per minute (TPM)")
assert ra.retry_after_seconds("Please try again in 12.5s.") == 14.5
assert ra.retry_after_seconds("Please try again in 7m20.208s.") == 180.0       # capped
assert ra.retry_after_seconds("no hint here") == 65.0
assert ra.retry_hint("Please try again in 7m20.208s. Need more tokens?") == "7m20.208s"
assert ra.retry_hint("Please try again in 2h3m10.5s.") == "2h3m10.5s" and ra.retry_hint("nothing") == "an unknown time"
res, tok = ra.invoke_with_usage(FakeGraph(), {"machine_id": "M1"})
assert tok is None or isinstance(tok, int)                                     # works with or without langchain installed

# report aggregation, on hand-made rows
def row(cid, kind, truth, score, conf, predicted, mode="failure_investigation"):
    return dict(case_id=cid, kind=kind, truth_raw=truth, score=score, final_confidence=conf, predicted=predicted,
                mode=mode, unverified_facts=0, candidates=[{"cause": "x", "confidence": "low", "capped": True}],
                retries=0, elapsed_s=10.0, n_history_readings=50)
rows = [row("inc-1", "incident", "Power Failure", "exact", "high", [H_POWER]),
        row("inc-2", "incident", "Tool Wear Failure", "overclaim", "medium", [H_SENSOR]),
        row("inc-3", "incident", "Tool Wear Failure", "exact", "medium", ["Tool wear failure"]),
        row("healthy-1", "healthy", "No Failure", "exact", "medium", [], "health_check")]
base = {"inc-1": "exact", "inc-2": "missed", "inc-3": "missed", "healthy-1": "overclaim"}
sm = summarize(rows, base)
assert sm["regressions"] == [] and sorted(sm["gains"]) == ["healthy-1", "inc-3"]
assert sm["confident_errors"] == ["inc-2"] and sm["labels"]["Tool Wear Failure"]["agent_exact"] == 1
assert sm["faith"]["healthy_in_health_mode"] == (1, 1)
text = format_report(sm)
assert "Regressions" in text and "confident errors" in text
lo, hi = wilson(60, 68); assert abs(lo - 0.785) < 0.002 and abs(hi - 0.939) < 0.002
# result-file names and the dev set
from evaluation.common import results_suffix
assert results_suffix() == "" and results_suffix("v1_pilot") == "_v1_pilot" and results_suffix(None, True, False) == "_dev"
assert results_suffix(None, False, True) == "_nomaint" and results_suffix("x", False, True) == "_x_nomaint"
assert len(ra.DEV_CASE_IDS) == 12 and "inc-117" in ra.DEV_CASE_IDS
print("ALL runner/report TESTS PASSED")
