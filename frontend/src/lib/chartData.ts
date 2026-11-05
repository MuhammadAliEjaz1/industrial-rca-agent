import type { IncidentRecord, SensorReading } from "../types";

export interface ChartPoint {
  i: number;
  timestamp: string;
  air: number;
  process: number;
  gap: number;
  rpm: number;
  torque: number;
  power: number;
  wear: number;
  incident?: string;
}

/** Readings (oldest first) + incidents -> chart rows. An incident's date equals its reading's timestamp. */
export function buildChartData(readings: SensorReading[], incidents: IncidentRecord[]): ChartPoint[] {
  const byTime = new Map(incidents.map((x) => [x.date, x.failure_type ?? "incident"]));
  return readings.map((r, i) => ({
    i: i + 1,
    timestamp: r.timestamp,
    air: r.air_temperature,
    process: r.process_temperature,
    gap: r.process_temperature - r.air_temperature,
    rpm: r.rotational_speed,
    torque: r.torque,
    power: (r.torque * r.rotational_speed * 2 * Math.PI) / 60,
    wear: r.tool_wear,
    incident: byTime.get(r.timestamp),
  }));
}
