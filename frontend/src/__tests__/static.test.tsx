import { renderHook, waitFor, act } from "@testing-library/react";
import { vi } from "vitest";
import { api, ApiError, REPLAY_STEP_MS, useInvestigation } from "../api";
import { parseRoute } from "../lib/route";

const entry = (step: string) => ({ step, status: "done", summary: `did ${step}`, timestamp: "2026-01-01T10:00:00+00:00" });
const recorded = {
  investigation_id: "inc-105", status: "done", machine_id: "M035", request: { machine_id: "M035", failure_reported: true },
  created_at: "2026-01-01T10:00:00+00:00", log: [entry("gather_context"), entry("run_anomaly_model"), entry("generate_report")],
  report: { most_supported_cause: "Tool wear failure", confidence_level: "medium" },
  demo: { kind: "incident", split: "test", recorded_label: "Tool Wear Failure", score: "exact" },
};
const index = [
  { investigation_id: "inc-105", machine_id: "M035", status: "done", created_at: "x" },
  { investigation_id: "healthy-14", machine_id: "M001", status: "done", created_at: "y" },
];

const files: Record<string, unknown> = {
  "/demo-data/machines.json": [{ machine_id: "M001" }],
  "/demo-data/investigations/index.json": index,
  "/demo-data/investigations/inc-105.json": recorded,
};
const fetchMock = vi.fn(async (url: string) => (url in files ? { ok: true, json: async () => files[url] } : { ok: false, status: 404, statusText: "Not Found", json: async () => ({}) }));

beforeEach(() => {
  vi.stubEnv("VITE_STATIC", "1");
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockClear();
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("static (recorded demo) data adapter", () => {
  it("reads JSON files instead of calling the backend", async () => {
    expect(await api.machines()).toEqual([{ machine_id: "M001" }]);
    expect(fetchMock).toHaveBeenCalledWith("/demo-data/machines.json");
  });

  it("lists only the recorded runs of the machine asked for", async () => {
    const runs = await api.investigations("M035");
    expect(runs.map((r) => r.investigation_id)).toEqual(["inc-105"]);
  });

  it("refuses to start a new investigation, and says why", async () => {
    await expect(api.start({ machine_id: "M001", failure_reported: true })).rejects.toThrow(/recorded demo/);
    await expect(api.start({ machine_id: "M001", failure_reported: true })).rejects.toBeInstanceOf(ApiError);
  });

  it("explains a missing file instead of crashing", async () => {
    await expect(api.machine("M999")).rejects.toThrow(/not part of the recorded demo/);
  });
});

describe("replay of a recorded run", () => {
  it("reveals the saved steps one by one, then the report", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { result } = renderHook(() => useInvestigation("inc-105"));
    await waitFor(() => expect(result.current.run?.status).toBe("running"));
    expect(result.current.run?.report).toBeNull();
    expect(result.current.steps).toHaveLength(0);

    await act(async () => { await vi.advanceTimersByTimeAsync(REPLAY_STEP_MS + 10); });
    expect(result.current.steps.map((s) => s.step)).toEqual(["gather_context"]);
    await act(async () => { await vi.advanceTimersByTimeAsync(REPLAY_STEP_MS * 2); });
    expect(result.current.steps).toHaveLength(3);
    expect(result.current.run?.status).toBe("running");               // the report only appears after the last step

    await act(async () => { await vi.advanceTimersByTimeAsync(REPLAY_STEP_MS + 10); });
    expect(result.current.run?.status).toBe("done");
    expect(result.current.run?.report?.most_supported_cause).toBe("Tool wear failure");
  });

  it("never opens a live stream", async () => {
    const created: unknown[] = [];
    vi.stubGlobal("EventSource", class { constructor() { created.push(this); } addEventListener() {} close() {} });
    renderHook(() => useInvestigation("inc-105"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(created).toHaveLength(0);
  });

  it("stops its timers when the component goes away", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { result, unmount } = renderHook(() => useInvestigation("inc-105"));
    await waitFor(() => expect(result.current.run?.status).toBe("running"));
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(REPLAY_STEP_MS * 10); });
    expect(result.current.steps).toHaveLength(0);
  });
});

describe("parseRoute", () => {
  it("understands every address the app uses", () => {
    expect(parseRoute("")).toEqual({ name: "fleet" });
    expect(parseRoute("#/")).toEqual({ name: "fleet" });
    expect(parseRoute("#/runs")).toEqual({ name: "runs" });
    expect(parseRoute("#/machine/M001")).toEqual({ name: "machine", id: "M001", run: null });
    expect(parseRoute("#/machine/M035/run/inc-105")).toEqual({ name: "machine", id: "M035", run: "inc-105" });
    expect(parseRoute("#/machine/M%2035")).toEqual({ name: "machine", id: "M 35", run: null });
  });
});
