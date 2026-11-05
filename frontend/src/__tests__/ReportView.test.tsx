import { render, screen, within } from "@testing-library/react";
import { ReportView } from "../components/ReportView";
import { Timeline } from "../components/Timeline";
import report from "../fixtures/inc105.json";
import type { InvestigationReport, LogEntry } from "../types";

const r = report as unknown as InvestigationReport; // a REAL report produced by the agent (case inc-105)

describe("ReportView on a real agent report", () => {
  it("keeps conclusion, hypotheses, observed facts and ML prediction in separate labelled blocks", () => {
    render(<ReportView report={r} />);
    expect(screen.getByText(/Conclusion · supported by the evidence/)).toBeInTheDocument();
    expect(screen.getByText(/Hypotheses · proposed by the AI model/)).toBeInTheDocument();
    expect(screen.getByText(/Observed facts · from the data/)).toBeInTheDocument();
    expect(screen.getByText(/ML prediction · a prediction, not a fact/)).toBeInTheDocument();
  });

  it("shows the verdict with its confidence", () => {
    const { container } = render(<ReportView report={r} />);
    const conclusion = container.querySelector(".block-conclusion") as HTMLElement;
    expect(within(conclusion).getByText("Tool wear failure")).toBeInTheDocument();
    expect(within(conclusion).getByText("MEDIUM CONFIDENCE")).toBeInTheDocument();
    expect(within(conclusion).getByText("Failure investigation")).toBeInTheDocument();
    expect(within(conclusion).getByText(/Classifier disagreement/)).toBeInTheDocument();
  });

  it("explains the confidence with the documented rule, not only the model's words", () => {
    render(<ReportView report={r} />);
    expect(screen.getAllByText(/Confidence basis \(documented rule\):/).length).toBe(2);
    // the same reason appears in the hypothesis card and in the code-written justification
    expect(screen.getAllByText(/weak signal \(documented recall/).length).toBe(2);
  });

  it("says when a hypothesis was settled by a rule without asking the model", () => {
    render(<ReportView report={r} />);
    expect(screen.getAllByText(/Not sent to the AI model/).length).toBe(1);
  });

  it("separates computed facts from model-written ones and hides the [computed] marker", () => {
    const { container } = render(<ReportView report={r} />);
    const observed = container.querySelector(".block-observed") as HTMLElement;
    expect(within(observed).getByText("Computed by code from the latest reading")).toBeInTheDocument();
    expect(within(observed).getByText(/Latest tool wear = 235 min/)).toBeInTheDocument();
    expect(within(observed).getByText(/Written by the AI model/)).toBeInTheDocument();
    expect(observed.textContent).not.toContain("[computed]");
  });

  it("shows the classifier without claiming it agrees", () => {
    const { container } = render(<ReportView report={r} />);
    const ml = container.querySelector(".block-ml") as HTMLElement;
    expect(within(ml).getByText(/PREDICTION: No Failure/)).toBeInTheDocument();
    expect(within(ml).getByText(/No reliability warning/)).toBeInTheDocument();
  });

  it("shows the recorded label of a replayed incident, and only when given", () => {
    const { rerender } = render(<ReportView report={r} />);
    expect(screen.queryByText(/Recorded outcome in the dataset/)).toBeNull();
    rerender(<ReportView report={r} recordedLabel="Tool Wear Failure" />);
    expect(screen.getByText(/Recorded outcome in the dataset/)).toBeInTheDocument();
    expect(screen.getByText(/never saw this label/)).toBeInTheDocument();
  });

  it("renders a maintenance advisory when there is one", () => {
    render(<ReportView report={{ ...r, advisories: ["Latest tool wear is 215 min ..."] }} />);
    expect(screen.getByText(/Maintenance advisory/)).toBeInTheDocument();
  });
});

describe("Timeline", () => {
  const step = (name: string): LogEntry => ({ step: name, status: "done", summary: `did ${name}`, timestamp: "2026-01-01T10:00:00+00:00" });

  it("marks finished steps and the next one as running", () => {
    const { container } = render(<Timeline steps={[step("gather_context"), step("run_anomaly_model")]} status="running" />);
    expect(container.querySelectorAll(".tl-done").length).toBe(2);
    expect(container.querySelector(".tl-running .tl-name")?.textContent).toBe("Collect incident evidence");
    expect(container.querySelectorAll(".tl-pending").length).toBe(4);
  });

  it("shows nothing as running once the run is over", () => {
    const all = ["gather_context", "run_anomaly_model", "gather_evidence", "generate_hypotheses", "investigate_hypotheses", "determine_root_cause", "generate_report"].map(step);
    const { container } = render(<Timeline steps={all} status="done" />);
    expect(container.querySelectorAll(".tl-done").length).toBe(7);
    expect(container.querySelector(".tl-running")).toBeNull();
  });
});
