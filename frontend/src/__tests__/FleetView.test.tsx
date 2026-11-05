import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import type { MachineSummary } from "../types";

vi.mock("../api", () => ({ api: { machines: vi.fn() } }));
import { api } from "../api";
import { FleetView } from "../views/FleetView";

const m = (id: string, status: MachineSummary["latest_status"]): MachineSummary => ({
  machine_id: id, attributes: { machine_type: "M", location: "Line A - Bay 1" }, n_readings: 290, latest_reading_at: "2026-08-31 13:34:27",
  n_incidents: 8, n_maintenance: 4, latest_status: status,
});

describe("FleetView", () => {
  beforeEach(() => {
    vi.mocked(api.machines).mockResolvedValue([m("M001", "ok"), m("M002", "limit_violated"), m("M003", "tool_wear_hint"), m("M004", "ok")]);
  });

  it("lists every machine and counts statuses", async () => {
    render(<FleetView />);
    await waitFor(() => expect(screen.getAllByRole("link")).toHaveLength(4));
    expect(screen.getByTestId("count-all")).toHaveTextContent("4");
    expect(screen.getByTestId("count-ok")).toHaveTextContent("2");
    expect(screen.getByTestId("count-limit_violated")).toHaveTextContent("1");
    expect(screen.getByTestId("count-tool_wear_hint")).toHaveTextContent("1");
  });

  it("links each card to its machine page", async () => {
    render(<FleetView />);
    const link = await screen.findByRole("link", { name: /M002/ });
    expect(link).toHaveAttribute("href", "#/machine/M002");
  });

  it("filters by status and by search text", async () => {
    render(<FleetView />);
    await screen.findAllByRole("link");
    fireEvent.click(screen.getByRole("button", { name: /Limit violated/ }));
    expect(screen.getAllByRole("link")).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: /Limit violated/ })); // toggles the filter off
    fireEvent.change(screen.getByLabelText("Find machine"), { target: { value: "m004" } });
    expect(screen.getAllByRole("link")).toHaveLength(1);
    expect(screen.getByRole("link", { name: /M004/ })).toBeInTheDocument();
  });

  it("shows a clear message when the API is unreachable", async () => {
    vi.mocked(api.machines).mockRejectedValue(new Error("Cannot reach the API at http://localhost:8000. Is the backend running?"));
    render(<FleetView />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/Cannot reach the API/);
  });
});
