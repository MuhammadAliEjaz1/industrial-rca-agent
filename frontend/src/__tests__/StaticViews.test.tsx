import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

vi.mock("../api", () => ({
  isStatic: () => true,
  useInvestigation: () => ({ run: null, steps: [], problem: null }),
  api: {
    runIndex: vi.fn(),
    investigations: vi.fn(),
  },
}));
import { api } from "../api";
import { InvestigatePanel } from "../components/InvestigatePanel";
import { RunsView } from "../views/RunsView";

const run = (id: string, machine: string, cause: string, score: string, kind = "incident", label = "Power Failure") => ({
  investigation_id: id, machine_id: machine, status: "done", created_at: "2026-01-01T10:00:00+00:00",
  most_supported_cause: cause, confidence_level: "high", demo: { kind, split: "test", recorded_label: label, score },
});

describe("RunsView (recorded demo)", () => {
  it("lists recorded runs with the agent's outcome, the recorded outcome and the match", async () => {
    vi.mocked(api.runIndex).mockResolvedValue([
      run("inc-18", "M012", "Power failure", "exact"),
      run("inc-72", "M020", "Overstrain failure", "wrong", "incident", "Tool Wear Failure"),
    ] as never);
    render(<RunsView />);
    expect(await screen.findByText("inc-18")).toBeInTheDocument();
    expect(screen.getByText(/2 runs of the frozen agent/)).toBeInTheDocument();
    expect(screen.getByText(/1 named the recorded outcome exactly/)).toBeInTheDocument();
    expect(screen.getByText("Tool Wear Failure")).toBeInTheDocument();             // the recorded outcome of inc-72
    expect(screen.getByText("wrong")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "inc-72" })).toHaveAttribute("href", "#/machine/M020/run/inc-72");
    expect(screen.getByText(/not a random one/)).toBeInTheDocument();             // the honest caveat is on the page
  });
});

describe("Investigate tab in the recorded demo", () => {
  it("offers replays, not a Run button", async () => {
    vi.mocked(api.investigations).mockResolvedValue([run("inc-18", "M012", "Power failure", "exact")] as never);
    render(<InvestigatePanel machineId="M012" incidents={[]} />);
    expect(await screen.findByRole("button", { name: "Replay" })).toBeInTheDocument();
    expect(screen.getByText(/no live AI here/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Run investigation/ })).toBeNull();
  });

  it("says so when a machine has no recorded run", async () => {
    vi.mocked(api.investigations).mockResolvedValue([]);
    render(<InvestigatePanel machineId="M099" incidents={[]} />);
    await waitFor(() => expect(screen.getByText(/No recorded investigation exists/)).toBeInTheDocument());
    expect(screen.getByRole("link", { name: /See all recorded runs/ })).toHaveAttribute("href", "#/runs");
  });
});
