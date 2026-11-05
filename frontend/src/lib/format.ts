import type { MachineStatus } from "../types";

export const STATUS_LABEL: Record<MachineStatus, string> = {
  ok: "Within limits",
  limit_violated: "Limit violated",
  tool_wear_hint: "Tool wear 200+ min",
  no_data: "No data",
};

export function fmt(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export type Tone = "high" | "medium" | "low" | "none";

export function confidenceTone(level: string | null | undefined): Tone {
  const l = (level ?? "").toLowerCase();
  return l === "high" || l === "medium" || l === "low" ? l : "none";
}

export function clockTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString("en-GB");
}

/** "[computed] Latest reading..." -> {computed: true, text: "Latest reading..."} */
export function splitFact(fact: string): { computed: boolean; text: string } {
  return fact.startsWith("[computed] ") ? { computed: true, text: fact.slice(11) } : { computed: false, text: fact };
}

export const STEP_ORDER = [
  "gather_context",
  "run_anomaly_model",
  "gather_evidence",
  "generate_hypotheses",
  "investigate_hypotheses",
  "determine_root_cause",
  "generate_report",
] as const;

export const STEP_LABEL: Record<string, string> = {
  gather_context: "Load sensor readings",
  run_anomaly_model: "Run ML classifier",
  gather_evidence: "Collect incident evidence",
  generate_hypotheses: "Generate hypotheses",
  investigate_hypotheses: "Test each hypothesis",
  determine_root_cause: "Determine outcome",
  generate_report: "Assemble report",
};
