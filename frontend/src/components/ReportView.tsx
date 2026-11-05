import { confidenceTone, fmt, splitFact } from "../lib/format";
import type { InvestigationReport } from "../types";
import { HypothesisCard } from "./HypothesisCard";
import { Badge, Block } from "./ui";

const MODE_LABEL: Record<string, string> = { health_check: "Health check", failure_investigation: "Failure investigation" };

export function ReportView({ report, recordedLabel }: { report: InvestigationReport; recordedLabel?: string | null }) {
  const chosen = new Set((report.most_supported_cause ?? "").split(" + "));
  const facts = report.observed_facts.map(splitFact);
  const computed = facts.filter((f) => f.computed);
  const reported = facts.filter((f) => !f.computed);
  const ml = report.ml_analysis;
  return (
    <div className="report">
      <Block title="Conclusion · supported by the evidence below" kind="conclusion">
        <div className="verdict">
          <span className="verdict-cause">{report.most_supported_cause ?? "No conclusion"}</span>
          <Badge tone={confidenceTone(report.confidence_level)}>{(report.confidence_level ?? "n/a").toUpperCase()} CONFIDENCE</Badge>
          {report.mode && <Badge tone="info">{MODE_LABEL[report.mode] ?? report.mode}</Badge>}
        </div>
        {report.advisories.map((a, i) => (
          <p key={i} className="advisory"><b>Maintenance advisory.</b> {a}</p>
        ))}
        {report.justification && <p>{report.justification}</p>}
        {recordedLabel && (
          <p className="recorded"><b>Recorded outcome in the dataset (replayed incident):</b> {recordedLabel}. The agent never saw this label.</p>
        )}
      </Block>

      <Block title="Hypotheses · proposed by the AI model, then tested" kind="hypothesis">
        <p className="muted small">The AI model chooses candidates from a fixed list of failure modes this system can test. Confidence is set by documented rules wherever a rule exists.</p>
        {report.candidate_root_causes.map((h) => (
          <HypothesisCard key={h.cause} h={h} chosen={chosen.has(h.cause) || h.cause === report.most_supported_cause} />
        ))}
      </Block>

      <Block title="Observed facts · from the data, not from the model" kind="observed">
        <h4>Computed by code from the latest reading</h4>
        <ul>{computed.map((f, i) => <li key={i}>{f.text}</li>)}</ul>
        {reported.length > 0 && (
          <>
            <h4>Written by the AI model (numbers are checked against the evidence)</h4>
            <ul>{reported.map((f, i) => <li key={i}>{f.text}</li>)}</ul>
          </>
        )}
      </Block>

      <Block title="ML prediction · a prediction, not a fact" kind="ml">
        <ul>{report.ml_predictions.map((p, i) => <li key={i}>{p}</li>)}</ul>
        {ml && (
          <p className="muted small">
            Classifier: {ml.predicted_class ?? "n/a"} ({fmt((ml.confidence ?? 0) * 100, 1)}%).
            {ml.reliability_warning ? ` Reliability warning: ${ml.reliability_note ?? ""}` : " No reliability warning for this class."}
          </p>
        )}
      </Block>

      {report.unsupported_speculation.length > 0 && (
        <Block title="Considered and not supported" kind="neutral">
          <ul>{report.unsupported_speculation.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </Block>
      )}

      {report.evidence_sources.length > 0 && (
        <Block title="Similar historical incidents consulted" kind="neutral">
          <p className="small muted">Similarity is by sensor profile among past failures only, so resemblance to a failure is not evidence that this machine is failing.</p>
          <ul>{report.evidence_sources.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </Block>
      )}
    </div>
  );
}
