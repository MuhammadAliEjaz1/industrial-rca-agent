import { useInvestigation } from "../api";
import { ErrorBox } from "./ui";
import { ReportView } from "./ReportView";
import { Timeline } from "./Timeline";

export function RunView({ runId, recordedLabel }: { runId: string; recordedLabel?: string | null }) {
  const { run, steps, problem } = useInvestigation(runId);
  const status = run?.status ?? null;
  const running = status !== "done" && status !== "error";
  return (
    <div className="runview">
      <div className="runhead">
        <span className="muted small">Investigation {runId}</span>
        {status === "queued" && steps.length === 0 && <span className="muted small"> · waiting in the queue (one investigation runs at a time)</span>}
        {running && steps.length > 0 && <span className="muted small"> · running</span>}
      </div>
      {problem && <ErrorBox message={problem} />}
      <Timeline steps={steps} status={status} />
      {status === "error" && <ErrorBox message={run?.error ?? "The investigation failed."} />}
      {status === "done" && run?.report && <ReportView report={run.report} recordedLabel={recordedLabel} />}
    </div>
  );
}
