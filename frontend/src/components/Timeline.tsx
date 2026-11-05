import { clockTime, STEP_LABEL, STEP_ORDER } from "../lib/format";
import type { LogEntry, RunStatus } from "../types";

export function Timeline({ steps, status }: { steps: LogEntry[]; status: RunStatus | null }) {
  const doneBy = new Map<string, LogEntry>();
  steps.forEach((s) => doneBy.set(s.step, s)); // the latest entry of a step wins
  const over = status === "done" || status === "error";
  const firstPending = STEP_ORDER.find((s) => !doneBy.has(s));
  return (
    <ol className="timeline" aria-label="Investigation timeline">
      {STEP_ORDER.map((name) => {
        const entry = doneBy.get(name);
        const state = entry ? "done" : name === firstPending && !over ? "running" : "pending";
        return (
          <li key={name} className={`tl tl-${state}`}>
            <span className="tl-dot" aria-hidden="true" />
            <div className="tl-body">
              <div className="tl-head">
                <span className="tl-name">{STEP_LABEL[name]}</span>
                {entry && <span className="muted small">{clockTime(entry.timestamp)}</span>}
                {state === "running" && <span className="muted small">working…</span>}
              </div>
              {entry && <div className="tl-summary muted small">{entry.summary}</div>}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
