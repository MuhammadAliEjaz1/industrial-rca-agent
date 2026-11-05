import { useEffect, useState } from "react";
import { api } from "../api";
import { clockTime } from "../lib/format";
import type { IncidentRecord, InvestigationSummary } from "../types";
import { Badge, ErrorBox } from "./ui";
import { RunView } from "./RunView";

export function InvestigatePanel({ machineId, incidents }: { machineId: string; incidents: IncidentRecord[] }) {
  const [reported, setReported] = useState(true);
  const [replay, setReplay] = useState<string>("");
  const [runId, setRunId] = useState<string | null>(null);
  const [recorded, setRecorded] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [past, setPast] = useState<InvestigationSummary[]>([]);

  const loadPast = () => api.investigations(machineId).then(setPast).catch(() => setPast([]));
  useEffect(() => {
    setRunId(null);
    setRecorded(null);
    void loadPast();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [machineId]);

  const chosen = incidents.find((i) => String(i.incident_id) === replay) ?? null;

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const body = chosen
        ? { machine_id: machineId, failure_reported: true, as_of: chosen.date, incident_id: chosen.incident_id }
        : { machine_id: machineId, failure_reported: reported };
      const accepted = await api.start(body);
      setRecorded(chosen ? chosen.failure_type ?? null : null);
      setRunId(accepted.investigation_id);
      void loadPast();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="form">
        <fieldset disabled={busy}>
          <legend>What happened?</legend>
          <label>
            <input type="radio" checked={!chosen && reported} disabled={!!chosen} onChange={() => setReported(true)} /> A failure was reported
          </label>
          <label>
            <input type="radio" checked={!chosen && !reported} disabled={!!chosen} onChange={() => setReported(false)} /> Routine check (nothing reported)
          </label>
          <label className="replay">
            Or replay a recorded incident
            <select value={replay} onChange={(e) => setReplay(e.target.value)}>
              <option value="">— none: investigate the latest reading —</option>
              {incidents.map((i) => (
                <option key={i.incident_id} value={i.incident_id}>
                  Incident {i.incident_id} · {i.date.slice(0, 16)}
                </option>
              ))}
            </select>
          </label>
          {chosen && <p className="muted small">The agent investigates as of that moment and never sees the recorded failure type or its own row; the recorded label is shown only after the report.</p>}
        </fieldset>
        <button className="btn" onClick={submit} disabled={busy}>{busy ? "Starting…" : "Run investigation"}</button>
        <p className="muted small">Each investigation uses AI tokens (about 2,000 to 8,000) and takes roughly 15 to 70 seconds. The demo allows a limited number per hour.</p>
        {error && <ErrorBox message={error} />}
      </div>

      {runId && <RunView key={runId} runId={runId} recordedLabel={recorded} />}

      {past.length > 0 && (
        <div className="past">
          <h4>Earlier investigations of this machine</h4>
          <ul>
            {past.map((p) => (
              <li key={p.investigation_id}>
                <button className="linklike" onClick={() => { setRunId(p.investigation_id); setRecorded(null); }}>
                  {clockTime(p.created_at)}
                </button>{" "}
                <Badge tone={p.status === "error" ? "low" : "info"}>{p.status}</Badge> {p.most_supported_cause ?? ""}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
