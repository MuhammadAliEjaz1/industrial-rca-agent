import type { ReactNode } from "react";
import { STATUS_LABEL, type Tone } from "../lib/format";
import type { MachineStatus } from "../types";

export function StatusLed({ status }: { status: MachineStatus }) {
  return <span className={`led led-${status}`} role="img" aria-label={STATUS_LABEL[status]} title={STATUS_LABEL[status]} />;
}

export function Badge({ tone, children, title }: { tone: Tone | "info" | "warn"; children: ReactNode; title?: string }) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  );
}

/** A report block with a coloured edge: observed (blue), ML (violet), hypothesis (amber), conclusion (green). */
export function Block({ title, kind, children }: { title: string; kind: "observed" | "ml" | "hypothesis" | "conclusion" | "neutral"; children: ReactNode }) {
  return (
    <section className={`block block-${kind}`}>
      <h3 className="block-title">{title}</h3>
      {children}
    </section>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="errorbox" role="alert">
      <strong>Something went wrong.</strong> {message}
      {onRetry && (
        <button className="btn btn-small" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
