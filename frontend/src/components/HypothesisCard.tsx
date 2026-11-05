import { confidenceTone } from "../lib/format";
import type { CandidateCause } from "../types";
import { Badge } from "./ui";

export function HypothesisCard({ h, chosen }: { h: CandidateCause; chosen: boolean }) {
  const byRule = h.evaluated_by === "rule";
  const disagrees = !!h.llm_confidence && h.llm_confidence !== h.confidence;
  return (
    <article className={`hyp ${chosen ? "hyp-chosen" : ""}`}>
      <header className="hyp-head">
        <span className="hyp-name">{h.cause}</span>
        <Badge tone={confidenceTone(h.confidence)}>{h.confidence.toUpperCase()}</Badge>
        {chosen && <Badge tone="info">IN CONCLUSION</Badge>}
      </header>
      {h.confidence_basis && (
        <p className="small">
          <b>Confidence basis (documented rule):</b> {h.confidence_basis}
        </p>
      )}
      {byRule && <p className="muted small">Not sent to the AI model: the documented rules already settle this hypothesis.</p>}
      {disagrees && (
        <p className="small note-warn">
          The AI model rated this <b>{h.llm_confidence}</b>; a documented rule set <b>{h.confidence}</b>. Both are kept so the
          disagreement is visible.
        </p>
      )}
      {!byRule && (
        <details>
          <summary>Evidence for and against</summary>
          <div className="evidence-cols">
            <div>
              <h4>Supports</h4>
              {h.supporting_evidence.length ? <ul>{h.supporting_evidence.map((e, i) => <li key={i}>{e}</li>)}</ul> : <p className="muted small">Nothing specific.</p>}
            </div>
            <div>
              <h4>Contradicts</h4>
              {h.contradicting_evidence.length ? <ul>{h.contradicting_evidence.map((e, i) => <li key={i}>{e}</li>)}</ul> : <p className="muted small">Nothing specific.</p>}
            </div>
          </div>
          {h.reasoning && <p className="small muted">{h.reasoning}</p>}
        </details>
      )}
    </article>
  );
}
