import { act, renderHook, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { useInvestigation } from "../api";

// A fake browser EventSource that the test can drive by hand.
class FakeES {
  static instances: FakeES[] = [];
  url: string;
  closed = false;
  listeners: Record<string, ((e: Event) => void)[]> = {};
  constructor(url: string) {
    this.url = url;
    FakeES.instances.push(this);
  }
  addEventListener(type: string, fn: (e: Event) => void) {
    (this.listeners[type] ??= []).push(fn);
  }
  close() {
    this.closed = true;
  }
  emit(type: string, ev: Event) {
    (this.listeners[type] ?? []).forEach((f) => f(ev));
  }
}

const entry = (step: string) => ({ step, status: "done", summary: `did ${step}`, timestamp: "2026-01-01T10:00:00+00:00" });
const run = (status: string, log: ReturnType<typeof entry>[], extra: object = {}) => ({
  investigation_id: "abc", status, machine_id: "M001", request: { machine_id: "M001", failure_reported: true },
  created_at: "2026-01-01T10:00:00+00:00", log, report: null, ...extra,
});

let current: unknown;
const fetchMock = vi.fn(async () => ({ ok: true, json: async () => current }));

beforeEach(() => {
  FakeES.instances = [];
  fetchMock.mockClear();
  vi.stubGlobal("EventSource", FakeES);
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("useInvestigation", () => {
  it("streams steps live, resuming after the steps already received, then loads the final report", async () => {
    current = run("running", [entry("gather_context")]);
    const { result } = renderHook(() => useInvestigation("abc"));
    await waitFor(() => expect(FakeES.instances).toHaveLength(1));
    expect(FakeES.instances[0].url).toContain("/investigations/abc/events?after=1");   // resumes after 1 known step
    expect(result.current.steps).toHaveLength(1);

    act(() => FakeES.instances[0].emit("step", new MessageEvent("step", { data: JSON.stringify(entry("run_anomaly_model")) })));
    expect(result.current.steps.map((s) => s.step)).toEqual(["gather_context", "run_anomaly_model"]);

    current = run("done", [entry("gather_context"), entry("run_anomaly_model")], { report: { most_supported_cause: "X" } });
    act(() => FakeES.instances[0].emit("done", new MessageEvent("done", { data: "{}" })));
    await waitFor(() => expect(result.current.run?.status).toBe("done"));
    expect(result.current.run?.report?.most_supported_cause).toBe("X");
    expect(FakeES.instances[0].closed).toBe(true);
  });

  it("does not open a stream for a run that is already finished", async () => {
    current = run("done", [entry("gather_context")], { report: { most_supported_cause: "Y" } });
    const { result } = renderHook(() => useInvestigation("abc"));
    await waitFor(() => expect(result.current.run?.status).toBe("done"));
    expect(FakeES.instances).toHaveLength(0);
  });

  it("shows the failure when the SERVER reports one (an 'error' event with data)", async () => {
    current = run("running", []);
    const { result } = renderHook(() => useInvestigation("abc"));
    await waitFor(() => expect(FakeES.instances).toHaveLength(1));
    current = run("error", [], { error: "The demo's daily AI token budget is used up.", error_kind: "daily_token_limit" });
    act(() => FakeES.instances[0].emit("error", new MessageEvent("error", { data: JSON.stringify({ status: "error" }) })));
    await waitFor(() => expect(result.current.run?.status).toBe("error"));
    expect(result.current.run?.error).toContain("token budget");
  });

  it("falls back to polling when the connection itself breaks", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    current = run("running", [entry("gather_context")]);
    const { result } = renderHook(() => useInvestigation("abc"));
    await waitFor(() => expect(FakeES.instances).toHaveLength(1));
    const callsBefore = fetchMock.mock.calls.length;

    act(() => FakeES.instances[0].emit("error", new Event("error")));      // a plain Event = connection problem
    current = run("done", [entry("gather_context")], { report: { most_supported_cause: "Z" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100);
    });
    await waitFor(() => expect(result.current.run?.status).toBe("done"));
    expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBefore);
    const callsAtDone = fetchMock.mock.calls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(fetchMock.mock.calls.length).toBe(callsAtDone);                 // polling stopped once the run was over
  });

  it("closes the stream when the component goes away", async () => {
    current = run("running", []);
    const { unmount } = renderHook(() => useInvestigation("abc"));
    await waitFor(() => expect(FakeES.instances).toHaveLength(1));
    unmount();
    expect(FakeES.instances[0].closed).toBe(true);
  });
});
