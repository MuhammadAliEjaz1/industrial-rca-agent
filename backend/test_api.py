"""
Tests for the API (no Groq, no real database): the real app is run over HTTP with a FAKE agent graph and a small
stand-in database. From backend/:   python test_api.py
"""
import json, sqlite3, tempfile, threading, time
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from api import db as apidb

# ---------------- a small stand-in database with the real table layout ----------------
tmp = Path(tempfile.mkdtemp())
DB = tmp / "industrial_rca.db"
c = sqlite3.connect(DB)
c.executescript("""
create table machines(machine_id text primary key, machine_type text, location text);
create table sensor_readings(reading_id integer primary key, machine_id text, timestamp text, air_temperature real,
  process_temperature real, rotational_speed real, torque real, tool_wear real);
create table maintenance_records(record_id integer primary key, machine_id text, date text, maintenance_type text,
  description text, technician text, parts_replaced text);
create table failure_incidents(incident_id integer primary key, machine_id text, date text, failure_type text,
  description text, resolution text, root_cause text);
""")
for m in ("M001", "M002", "M003", "M004"):
    c.execute("insert into machines values (?,?,?)", (m, "L", "Line 1"))
def reading(m, ts, air, proc, rpm, torque, wear):
    c.execute("insert into sensor_readings(machine_id,timestamp,air_temperature,process_temperature,rotational_speed,torque,tool_wear) values (?,?,?,?,?,?,?)",
              (m, ts, air, proc, rpm, torque, wear))
for i in range(5):                                               # M001: healthy, 5 readings
    reading("M001", f"2025-03-0{i + 1} 08:00:00.000001", 298.6, 308.3, 1589, 38.0, 100 + i)
reading("M002", "2025-03-01 08:00:00.000001", 300.0, 308.0, 1300, 40.0, 50)           # M002: heat dissipation limit met
reading("M004", "2025-03-01 08:00:00.000001", 298.0, 308.3, 1500, 30.0, 215)          # M004: tool wear 215 (hint only)
# M003: no readings at all
c.execute("insert into maintenance_records(machine_id,date,maintenance_type,description,technician,parts_replaced) values ('M001','2025-03-02 00:00:00','Scheduled','Coolant inspection','T. Ahmed',NULL)")
c.execute("insert into maintenance_records(machine_id,date,maintenance_type,description,technician,parts_replaced) values ('M001','2025-03-09 00:00:00','Reactive','Later record','R. Hussain','Tool')")
c.execute("insert into failure_incidents values (1,'M001','2025-03-03 08:00:00.000001','Power Failure','Machine failure detected: Power Failure.',NULL,'Power Failure')")
c.execute("insert into failure_incidents values (2,'M001','2025-03-05 08:00:00.000001','Heat Dissipation Failure','d',NULL,'Heat Dissipation Failure')")
c.commit(); c.close()

# ---------------- a fake agent graph with the same .stream() contract as LangGraph ----------------
def step(name, text): return {"step": name, "status": "done", "summary": text, "timestamp": "2026-01-01T00:00:00+00:00"}
REPORT = {"mode": "failure_investigation", "most_supported_cause": "Heat dissipation failure", "confidence_level": "high",
          "justification": "j", "observed_facts": ["f"], "ml_predictions": ["m"], "advisories": [], "unsupported_speculation": [],
          "candidate_root_causes": [{"cause": "Heat dissipation failure", "confidence": "high", "supporting_evidence": ["s"],
                                     "contradicting_evidence": [], "reasoning": "r", "llm_confidence": "low", "confidence_basis": "b"}],
          "evidence_sources": [], "limit_checks": {"violated": ["heat_dissipation"]}, "ml_analysis": {"predicted_class": "X"}}

class FakeGraph:
    inputs, running, max_running, lock = [], 0, 0, threading.Lock()
    def __init__(self, error=None, delay=0.0): self.error, self.delay = error, delay
    def stream(self, state, stream_mode):
        assert stream_mode == "updates"
        with FakeGraph.lock:
            FakeGraph.inputs.append(state); FakeGraph.running += 1; FakeGraph.max_running = max(FakeGraph.max_running, FakeGraph.running)
        try:
            yield {"gather_context": {"sensor_readings": [1], "investigation_log": [step("gather_context", "Loaded 20 readings")]}}
            time.sleep(self.delay)
            if self.error:
                raise RuntimeError(self.error)
            yield {"run_anomaly_model": {"anomaly_result": {}, "investigation_log": [step("run_anomaly_model", "Predicted: X")]}}
            yield {"generate_hypotheses": {"hypotheses": [], "investigation_log": [step("generate_hypotheses", "Hypotheses: A, B")]}}
            yield {"generate_report": {"final_report": REPORT, "investigation_log": [step("generate_report", "Report assembled.")]}}
        finally:
            with FakeGraph.lock:
                FakeGraph.running -= 1

def make_client(graph=None, **kw):
    return TestClient(create_app(graph_factory=lambda: graph or FakeGraph(), db_path=DB, runs_dir=kw.pop("runs_dir", tmp / "runs"), **kw))

def wait_done(client, rid, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/investigations/{rid}").json()
        if r["status"] in ("done", "error"):
            return r
        time.sleep(0.05)
    raise AssertionError("investigation did not finish")

client = make_client()

# ---------------- general ----------------
assert client.get("/health").json() == {"status": "ok"}
assert any("SYNTHETIC" in n for n in client.get("/about").json()["notices"])

# ---------------- machines ----------------
ms = {m["machine_id"]: m for m in client.get("/machines").json()}
assert list(ms) == ["M001", "M002", "M003", "M004"]
assert ms["M001"]["n_readings"] == 5 and ms["M001"]["n_incidents"] == 2 and ms["M001"]["n_maintenance"] == 2
assert ms["M001"]["attributes"] == {"machine_type": "L", "location": "Line 1"}          # unknown columns pass through
assert (ms["M001"]["latest_status"], ms["M002"]["latest_status"], ms["M003"]["latest_status"], ms["M004"]["latest_status"]) == \
       ("ok", "limit_violated", "no_data", "tool_wear_hint")
d = client.get("/machines/M002").json()
assert d["limit_status"]["heat_dissipation"] and d["limit_status"]["hard_limit_violated"] and d["latest_reading"]["rotational_speed"] == 1300
assert client.get("/machines/M003").json()["latest_reading"] is None
for path in ("/machines/NOPE", "/machines/NOPE/sensors", "/machines/NOPE/maintenance", "/machines/NOPE/incidents"):
    assert client.get(path).status_code == 404, path

# sensors: chart order (oldest first), limit, as_of, validation
s = client.get("/machines/M001/sensors").json()
assert s["count"] == 5 and [r["tool_wear"] for r in s["readings"]] == [100, 101, 102, 103, 104] and s["notices"]
assert [r["tool_wear"] for r in client.get("/machines/M001/sensors?limit=2").json()["readings"]] == [103, 104]   # the 2 newest, oldest first
assert client.get("/machines/M001/sensors", params={"as_of": "2025-03-03 08:00:00.000001"}).json()["count"] == 3
assert client.get("/machines/M001/sensors?limit=0").status_code == 422
assert client.get("/machines/M001/maintenance").json()["count"] == 2
assert client.get("/machines/M001/maintenance", params={"as_of": "2025-03-05 00:00:00"}).json()["count"] == 1
inc = client.get("/machines/M001/incidents").json()
assert inc["count"] == 2 and inc["incidents"][0]["incident_id"] == 2                     # newest first
assert client.get("/machines/M001/incidents", params={"as_of": "2025-03-05 08:00:00.000001"}).json()["count"] == 1

# the API can never write to the database
conn = apidb._connect(DB)
try:
    conn.execute("insert into machines values ('X','L','y')"); raise SystemExit("write must fail")
except sqlite3.OperationalError as e:
    assert "readonly" in str(e).lower(), e

# ---------------- investigations ----------------
assert client.post("/investigations", json={"machine_id": "NOPE"}).status_code == 404
assert client.post("/investigations", json={"machine_id": "M002", "as_of": "yesterday"}).status_code == 422
assert client.post("/investigations", json={"machine_id": ""}).status_code == 422
FakeGraph.inputs.clear()
acc = client.post("/investigations", json={"machine_id": "M002", "as_of": "2025-03-01T08:00:00.000001", "incident_id": 7})
assert acc.status_code == 202
a = acc.json(); rid = a["investigation_id"]
assert a["status_url"] == f"/investigations/{rid}" and a["events_url"].endswith("/events")
done = wait_done(client, rid)
assert done["status"] == "done" and [l["step"] for l in done["log"]] == ["gather_context", "run_anomaly_model", "generate_hypotheses", "generate_report"]
assert done["report"]["most_supported_cause"] == "Heat dissipation failure" and done["report"]["limit_checks"]["violated"] == ["heat_dissipation"]
assert done["report"]["candidate_root_causes"][0]["confidence_basis"] == "b" and done["started_at"] and done["finished_at"]
# what the agent was given: normalised timestamp, the incident to hide, a neutral description, failure_reported defaults to True
inp = FakeGraph.inputs[-1]
assert inp["as_of"] == "2025-03-01 08:00:00.000001" and inp["incident_id"] == 7 and inp["failure_reported"] is True and inp["retry_count"] == 0
assert "Power Failure" not in inp["incident_description"] and "failure was reported" in inp["incident_description"]
routine = client.post("/investigations", json={"machine_id": "M001", "failure_reported": False}).json()
wait_done(client, routine["investigation_id"])
assert FakeGraph.inputs[-1]["failure_reported"] is False and "Routine" in FakeGraph.inputs[-1]["incident_description"]

# live timeline (Server-Sent Events): every step arrives, then a final 'done'
def read_sse(url):
    events, current = [], {}
    with client.stream("GET", url) as r:
        assert r.status_code == 200 and "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line.startswith(":"):
                continue
            if line == "":
                if current: events.append(current); current = {}
            else:
                k, _, v = line.partition(": "); current[k] = v
    return events
ev = read_sse(f"/investigations/{rid}/events")
assert [e["event"] for e in ev] == ["step", "step", "step", "step", "done"], ev
assert json.loads(ev[0]["data"])["step"] == "gather_context" and ev[0]["id"] == "1" and json.loads(ev[-1]["data"])["status"] == "done"
ev2 = read_sse(f"/investigations/{rid}/events?after=2")                       # resume after 2 steps
assert [e["event"] for e in ev2] == ["step", "step", "done"] and ev2[0]["id"] == "3"
assert client.get("/investigations/nope/events").status_code == 404 and client.get("/investigations/nope").status_code == 404

# a live run streams steps while it is still running
slow = make_client(FakeGraph(delay=0.4))
rid2 = slow.post("/investigations", json={"machine_id": "M001"}).json()["investigation_id"]
seen_running = False
with slow.stream("GET", f"/investigations/{rid2}/events") as r:
    first = None
    for line in r.iter_lines():
        if line.startswith("event: step") and first is None:
            first = time.time()
            seen_running = slow.get(f"/investigations/{rid2}").json()["status"] == "running"
assert seen_running, "the first step must arrive while the run is still going"

# failures: friendly message, never the raw exception; the worker keeps working afterwards
bad = make_client(FakeGraph(error="RuntimeError: boom with secret=abc"))
e1 = wait_done(bad, bad.post("/investigations", json={"machine_id": "M001"}).json()["investigation_id"])
assert e1["status"] == "error" and e1["error_kind"] == "internal" and "secret" not in e1["error"] and e1["report"] is None
tpd = make_client(FakeGraph(error="Error 429 ... tokens per day (TPD): Limit 200000, Used 198756"))
e2 = wait_done(tpd, tpd.post("/investigations", json={"machine_id": "M001"}).json()["investigation_id"])
assert e2["error_kind"] == "daily_token_limit" and "budget" in e2["error"]
ev3 = read_sse(f"/investigations/{e2['investigation_id']}/events")
assert ev3[-1]["event"] == "error" and json.loads(ev3[-1]["data"])["error_kind"] == "daily_token_limit"

# hourly budget protects the API key
capped = make_client(max_per_hour=2)
assert [capped.post("/investigations", json={"machine_id": "M001"}).status_code for _ in range(3)] == [202, 202, 429]

# one investigation at a time (Groq's free limits), the rest wait in the queue
FakeGraph.running = FakeGraph.max_running = 0
q = make_client(FakeGraph(delay=0.15))
ids = [q.post("/investigations", json={"machine_id": "M001"}).json()["investigation_id"] for _ in range(3)]
for i in ids:
    wait_done(q, i, timeout=15)
assert FakeGraph.max_running == 1, FakeGraph.max_running

# listing, and results that survive a restart
lst = client.get("/investigations", params={"machine_id": "M002"}).json()
assert [x["investigation_id"] for x in lst] == [rid] and lst[0]["most_supported_cause"] == "Heat dissipation failure"
restarted = make_client()                                                       # new app, same runs directory
assert restarted.get(f"/investigations/{rid}").json()["report"]["confidence_level"] == "high"
print("ALL API TESTS PASSED")
