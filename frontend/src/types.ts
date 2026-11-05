// Mirrors backend/api/models.py

export type MachineStatus = "ok" | "limit_violated" | "tool_wear_hint" | "no_data";

export interface SensorReading {
  timestamp: string;
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
}

export interface LimitStatus {
  temp_diff_k: number;
  power_w: number;
  wear_x_torque: number;
  heat_dissipation: boolean;
  power_out_of_range: boolean;
  possible_overstrain: boolean;
  tool_wear_at_failure_range: boolean;
  hard_limit_violated: boolean;
}

export interface MachineSummary {
  machine_id: string;
  attributes: Record<string, unknown>;
  n_readings: number;
  latest_reading_at: string | null;
  n_incidents: number;
  n_maintenance: number;
  latest_status: MachineStatus;
}

export interface MachineDetail extends MachineSummary {
  latest_reading: SensorReading | null;
  limit_status: LimitStatus | null;
}

export interface SensorSeries {
  machine_id: string;
  count: number;
  readings: SensorReading[];
  notices: string[];
}

export interface MaintenanceRecord {
  date: string;
  maintenance_type?: string | null;
  description?: string | null;
  technician?: string | null;
  parts_replaced?: string | null;
}

export interface IncidentRecord {
  incident_id: number;
  date: string;
  failure_type?: string | null;
  description?: string | null;
  resolution?: string | null;
  root_cause?: string | null;
}

export type RunStatus = "queued" | "running" | "done" | "error";

export interface LogEntry {
  step: string;
  status: string;
  summary: string;
  timestamp: string;
}

export interface CandidateCause {
  cause: string;
  confidence: string;
  supporting_evidence: string[];
  contradicting_evidence: string[];
  reasoning?: string | null;
  llm_confidence?: string | null;
  confidence_basis?: string | null;
  evaluated_by?: string | null;
}

export interface InvestigationReport {
  mode?: string | null;
  most_supported_cause?: string | null;
  confidence_level?: string | null;
  justification?: string | null;
  observed_facts: string[];
  ml_predictions: string[];
  advisories: string[];
  unsupported_speculation: string[];
  candidate_root_causes: CandidateCause[];
  evidence_sources: string[];
  limit_checks?: { latest?: Record<string, number | boolean> | null } | null;
  ml_analysis?: {
    predicted_class?: string | null;
    confidence?: number | null;
    reliability_warning?: boolean | null;
    reliability_note?: string | null;
  } | null;
}

export interface InvestigationRequest {
  machine_id: string;
  failure_reported: boolean;
  description?: string | null;
  as_of?: string | null;
  incident_id?: number | null;
}

export interface InvestigationAccepted {
  investigation_id: string;
  status: RunStatus;
  machine_id: string;
  status_url: string;
  events_url: string;
}

export interface InvestigationStatus {
  investigation_id: string;
  status: RunStatus;
  machine_id: string;
  request: InvestigationRequest;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  log: LogEntry[];
  report?: InvestigationReport | null;
  error?: string | null;
  error_kind?: string | null;
}

export interface InvestigationSummary {
  investigation_id: string;
  status: RunStatus;
  machine_id: string;
  created_at: string;
  finished_at?: string | null;
  most_supported_cause?: string | null;
  confidence_level?: string | null;
}
