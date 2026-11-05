import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { StatusLed, ErrorBox } from "../components/ui";
import { STATUS_LABEL } from "../lib/format";
import type { MachineStatus, MachineSummary } from "../types";

const ORDER: MachineStatus[] = ["limit_violated", "tool_wear_hint", "ok", "no_data"];

export function FleetView() {
  const [machines, setMachines] = useState<MachineSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<MachineStatus | "all">("all");
  const [query, setQuery] = useState("");

  const load = useCallback(() => {
    setError(null);
    api.machines().then(setMachines).catch((e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const counts = useMemo(() => {
    const c: Record<MachineStatus, number> = { ok: 0, limit_violated: 0, tool_wear_hint: 0, no_data: 0 };
    (machines ?? []).forEach((m) => (c[m.latest_status] += 1));
    return c;
  }, [machines]);

  if (error) return <ErrorBox message={error} onRetry={load} />;
  if (!machines) return <p className="muted">Loading machines…</p>;

  const shown = machines.filter(
    (m) => (filter === "all" || m.latest_status === filter) && m.machine_id.toLowerCase().includes(query.trim().toLowerCase()),
  );

  return (
    <div>
      <div className="summary-strip">
        <div className="sum"><span className="sum-n" data-testid="count-all">{machines.length}</span><span className="muted small">machines</span></div>
        {ORDER.map((s) => (
          <button key={s} className={`sum sum-btn ${filter === s ? "sum-active" : ""}`} onClick={() => setFilter(filter === s ? "all" : s)} aria-pressed={filter === s}>
            <span className="sum-n" data-testid={`count-${s}`}>{counts[s]}</span>
            <span className="small"><StatusLed status={s} /> {STATUS_LABEL[s]}</span>
          </button>
        ))}
        <input className="search" placeholder="Find machine (e.g. M012)" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Find machine" />
      </div>
      <p className="muted small">Status is a rule check of each machine's latest reading against the documented limits. It is not a prediction.</p>
      <div className="grid">
        {shown.map((m) => (
          <a key={m.machine_id} className="card" href={`#/machine/${encodeURIComponent(m.machine_id)}`}>
            <div className="card-top">
              <StatusLed status={m.latest_status} />
              <span className="card-id">{m.machine_id}</span>
              {typeof m.attributes.machine_type === "string" && <span className="chip">type {m.attributes.machine_type}</span>}
            </div>
            <div className="muted small">{String(m.attributes.location ?? "")}</div>
            <div className="card-stats small">
              <span>{m.n_readings} readings</span>
              <span>{m.n_incidents} recorded failures</span>
            </div>
            <div className="muted small">{STATUS_LABEL[m.latest_status]}</div>
          </a>
        ))}
        {shown.length === 0 && <p className="muted">No machine matches.</p>}
      </div>
    </div>
  );
}
