import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, ErrorBox } from "../components/ui";
import { confidenceTone } from "../lib/format";
import type { InvestigationSummary } from "../types";

const KIND: Record<string, string> = { incident: "Recorded failure", healthy: "Routine check", healthy_wear: "Routine check (wear 200+ min)" };
const SCORE_TONE: Record<string, "high" | "medium" | "low" | "info"> = { exact: "high", partial: "medium", wrong: "low", missed: "low", overclaim: "low" };

export function RunsView() {
  const [runs, setRuns] = useState<InvestigationSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.runIndex().then(setRuns).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorBox message={error} />;
  if (!runs) return <p className="muted">Loading recorded runs…</p>;
  const exact = runs.filter((r) => r.demo?.score === "exact").length;

  return (
    <div>
      <h2>Recorded investigations</h2>
      <p className="muted">
        {runs.length} runs of the frozen agent from the Phase 8 test run ({exact} named the recorded outcome exactly). This is a partial,
        stratified sample (37 of 106 planned cases), not a random one, so it says nothing about overall accuracy. Open a run to replay it.
      </p>
      <table className="data">
        <thead>
          <tr><th>Case</th><th>Machine</th><th>Situation</th><th>Agent's outcome</th><th>Recorded outcome</th><th>Match</th></tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.investigation_id}>
              <td><a href={`#/machine/${encodeURIComponent(r.machine_id)}/run/${encodeURIComponent(r.investigation_id)}`}>{r.investigation_id}</a></td>
              <td>{r.machine_id}</td>
              <td>{KIND[r.demo?.kind ?? ""] ?? "–"}</td>
              <td>{r.most_supported_cause} <Badge tone={confidenceTone(r.confidence_level)}>{(r.confidence_level ?? "n/a").toUpperCase()}</Badge></td>
              <td>{r.demo?.recorded_label ?? "–"}</td>
              <td><Badge tone={SCORE_TONE[r.demo?.score ?? ""] ?? "info"}>{r.demo?.score ?? "–"}</Badge></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
