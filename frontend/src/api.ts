import { useEffect, useState } from "react";
import type {
  IncidentRecord, InvestigationAccepted, InvestigationRequest, InvestigationStatus, InvestigationSummary,
  LogEntry, MachineDetail, MachineSummary, MaintenanceRecord, RunStatus, SensorSeries,
} from "./types";

export const API_BASE: string = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, init);
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_BASE}. Is the backend running?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* keep the status text */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  machines: () => request<MachineSummary[]>("/machines"),
  machine: (id: string) => request<MachineDetail>(`/machines/${enc(id)}`),
  sensors: (id: string, limit = 300) => request<SensorSeries>(`/machines/${enc(id)}/sensors?limit=${limit}`),
  maintenance: (id: string) => request<{ records: MaintenanceRecord[] }>(`/machines/${enc(id)}/maintenance`),
  incidents: (id: string) => request<{ incidents: IncidentRecord[] }>(`/machines/${enc(id)}/incidents`),
  start: (body: InvestigationRequest) =>
    request<InvestigationAccepted>("/investigations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  investigation: (id: string) => request<InvestigationStatus>(`/investigations/${enc(id)}`),
  investigations: (machineId: string) =>
    request<InvestigationSummary[]>(`/investigations?machine_id=${enc(machineId)}`),
  eventsUrl: (id: string, after: number) => `${API_BASE}/investigations/${enc(id)}/events?after=${after}`,
};

const isOver = (s: RunStatus) => s === "done" || s === "error";

/** Follows one investigation: live steps over Server-Sent Events, falling back to polling if the stream breaks. */
export function useInvestigation(id: string | null) {
  const [run, setRun] = useState<InvestigationStatus | null>(null);
  const [steps, setSteps] = useState<LogEntry[]>([]);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    setRun(null);
    setSteps([]);
    setProblem(null);
    if (!id) return;
    let closed = false;
    let es: EventSource | null = null;
    let timer: number | undefined;

    const refresh = async (): Promise<InvestigationStatus | null> => {
      try {
        const r = await api.investigation(id);
        if (!closed) {
          setRun(r);
          setSteps(r.log);
        }
        return r;
      } catch (e) {
        if (!closed) setProblem(e instanceof Error ? e.message : String(e));
        return null;
      }
    };
    const startPolling = () => {
      if (timer !== undefined) return;
      timer = window.setInterval(async () => {
        const r = await refresh();
        if (r && isOver(r.status) && timer !== undefined) {
          window.clearInterval(timer);
          timer = undefined;
        }
      }, 2000);
    };

    void refresh().then((r) => {
      if (closed || !r || isOver(r.status)) return;
      es = new EventSource(api.eventsUrl(id, r.log.length));
      es.addEventListener("step", (ev) => {
        const entry = JSON.parse((ev as MessageEvent).data) as LogEntry;
        setSteps((prev) => [...prev, entry]);
      });
      es.addEventListener("done", () => {
        es?.close();
        void refresh();
      });
      es.addEventListener("error", (ev) => {
        es?.close();
        if (ev instanceof MessageEvent) void refresh(); // the server reported a failed run
        else startPolling(); // a connection problem: fall back to polling
      });
    });

    return () => {
      closed = true;
      es?.close();
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [id]);

  return { run, steps, problem };
}
