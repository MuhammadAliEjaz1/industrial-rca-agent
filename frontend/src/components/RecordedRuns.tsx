import { useEffect, useState } from "react";
import { api } from "../api";
import { clockTime, confidenceTone } from "../lib/format";
import type { InvestigationSummary } from "../types";
import { Badge, ErrorBox } from "./ui";
import { RunView } from "./RunView";

const KIND_LABEL: Record<string, string> = { incident: "recorded failure", healthy: "routine check", healthy_wear: "routine check, tool wear 200+ min" };

/** The static demo's replacement for the "run investigation" form. */
export function RecordedRuns({ machineId, initialRunId }: { machineId: string; initialRunId?: string | null }) {
  const [runs, setRuns] = useState<InvestigationSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(initialRunId ?? null);

  useEffect(() => {
    setRuns(null);
    setError(null);
    api.investigations(machineId).then(setRuns).catch((e: Error) => setError(e.message));
  }, [machineId]);

  const open = runs?.find((r) => r.investigation_id === openId) ?? null;

  if (error) return <ErrorBox message={error} />;
  if (!runs) return <p className="muted">Loading recorded runs…</p>;

  return (
    <div>
      <p className="muted small">
        This is a <b>recorded demo</b>: there is no live AI here. The runs below were produced earlier by the real agent and are replayed step by step.
      </p>
      {runs.length === 0 ? (
        <p>No recorded investigation exists for this machine. <a href="#/runs">See all recorded runs</a>.</p>
      ) : (
        <ul className="runlist">
          {runs.map((r) => (
            <li key={r.investigation_id}>
              <button className="btn btn-small" onClick={() => setOpenId(r.investigation_id)} aria-pressed={r.investigation_id === openId}>
                Replay
              </button>{" "}
              <b>{r.investigation_id}</b> · {KIND_LABEL[r.demo?.kind ?? ""] ?? "run"} · {r.most_supported_cause ?? "no conclusion"}{" "}
              <Badge tone={confidenceTone(r.confidence_level)}>{(r.confidence_level ?? "n/a").toUpperCase()}</Badge>{" "}
              <span className="muted small">{clockTime(r.created_at)}</span>
            </li>
          ))}
        </ul>
      )}
      {open && (
        <RunView key={open.investigation_id} runId={open.investigation_id} recordedLabel={open.demo?.kind === "incident" ? open.demo.recorded_label : null} />
      )}
    </div>
  );
}
