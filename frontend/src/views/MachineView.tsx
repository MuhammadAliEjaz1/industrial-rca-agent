import { useEffect, useState } from "react";
import { api } from "../api";
import { InvestigatePanel } from "../components/InvestigatePanel";
import { LimitPanel } from "../components/LimitPanel";
import { SensorCharts } from "../components/SensorCharts";
import { ErrorBox, StatusLed } from "../components/ui";
import { STATUS_LABEL } from "../lib/format";
import type { IncidentRecord, MachineDetail, MaintenanceRecord, SensorReading } from "../types";

type Tab = "monitor" | "investigate" | "history";
interface Loaded { detail: MachineDetail; readings: SensorReading[]; incidents: IncidentRecord[]; maintenance: MaintenanceRecord[] }

export function MachineView({ machineId, openRun }: { machineId: string; openRun?: string | null }) {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>(openRun ? "investigate" : "monitor");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    Promise.all([api.machine(machineId), api.sensors(machineId, 300), api.incidents(machineId), api.maintenance(machineId)])
      .then(([detail, s, inc, m]) => {
        if (!cancelled) setData({ detail, readings: s.readings, incidents: inc.incidents, maintenance: m.records });
      })
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [machineId, attempt]);

  if (error) return <ErrorBox message={error} onRetry={() => setAttempt((a) => a + 1)} />;
  if (!data) return <p className="muted">Loading {machineId}…</p>;
  const { detail } = data;

  return (
    <div>
      <a href="#/" className="back">← All machines</a>
      <header className="machine-head">
        <StatusLed status={detail.latest_status} />
        <h2>{detail.machine_id}</h2>
        <span className="muted">{STATUS_LABEL[detail.latest_status]}</span>
        <span className="attrs small muted">
          {Object.entries(detail.attributes).map(([k, v]) => `${k.replace("_", " ")}: ${String(v)}`).join(" · ")}
        </span>
      </header>
      <nav className="tabs" role="tablist">
        {(["monitor", "investigate", "history"] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "tab tab-active" : "tab"} onClick={() => setTab(t)}>
            {t === "monitor" ? "Monitor" : t === "investigate" ? "Investigate" : "History"}
          </button>
        ))}
      </nav>

      {tab === "monitor" && (
        <>
          <section className="panel"><h3>Latest reading and limit checks</h3><LimitPanel reading={detail.latest_reading} status={detail.limit_status} /></section>
          <section className="panel"><h3>Sensor channels (last {data.readings.length} readings)</h3><SensorCharts readings={data.readings} incidents={data.incidents} /></section>
        </>
      )}
      {tab === "investigate" && <section className="panel"><h3>Investigate</h3><InvestigatePanel machineId={machineId} incidents={data.incidents} initialRunId={openRun} /></section>}
      {tab === "history" && (
        <>
          <section className="panel">
            <h3>Recorded failures</h3>
            {data.incidents.length === 0 ? <p className="muted">None.</p> : (
              <table className="data"><thead><tr><th>#</th><th>Time (synthetic)</th><th>Type</th></tr></thead>
                <tbody>{data.incidents.map((i) => <tr key={i.incident_id}><td>{i.incident_id}</td><td>{i.date.slice(0, 19)}</td><td>{i.failure_type}</td></tr>)}</tbody></table>
            )}
          </section>
          <section className="panel">
            <h3>Maintenance records</h3>
            <p className="muted small">These records are randomly generated demo data. The agent does not use them as evidence.</p>
            {data.maintenance.length === 0 ? <p className="muted">None.</p> : (
              <table className="data"><thead><tr><th>Date</th><th>Type</th><th>Description</th></tr></thead>
                <tbody>{data.maintenance.map((m, i) => <tr key={i}><td>{m.date.slice(0, 10)}</td><td>{m.maintenance_type}</td><td>{m.description}</td></tr>)}</tbody></table>
            )}
          </section>
        </>
      )}
    </div>
  );
}
