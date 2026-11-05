import { fmt } from "../lib/format";
import { LIMITS } from "../lib/limits";
import type { LimitStatus, SensorReading } from "../types";
import { Badge } from "./ui";

function Row({ label, value, rule, state }: { label: string; value: string; rule: string; state: "ok" | "hit" | "hint" }) {
  return (
    <tr>
      <td>{label}</td>
      <td className="num">{value}</td>
      <td className="muted small">{rule}</td>
      <td>{state === "hit" ? <Badge tone="low">LIMIT MET</Badge> : state === "hint" ? <Badge tone="warn">WEAK SIGNAL</Badge> : <Badge tone="info">OK</Badge>}</td>
    </tr>
  );
}

export function LimitPanel({ reading, status }: { reading: SensorReading | null; status: LimitStatus | null }) {
  if (!reading || !status) return <p className="muted">No readings yet.</p>;
  return (
    <div>
      <div className="reading-strip">
        <div><span className="muted small">Air</span><b>{fmt(reading.air_temperature)} K</b></div>
        <div><span className="muted small">Process</span><b>{fmt(reading.process_temperature)} K</b></div>
        <div><span className="muted small">Speed</span><b>{fmt(reading.rotational_speed, 0)} rpm</b></div>
        <div><span className="muted small">Torque</span><b>{fmt(reading.torque)} Nm</b></div>
        <div><span className="muted small">Tool wear</span><b>{fmt(reading.tool_wear, 0)} min</b></div>
      </div>
      <table className="limits">
        <tbody>
          <Row label="Heat dissipation" value={`gap ${fmt(status.temp_diff_k)} K`} rule={`fails when gap < ${LIMITS.heatGapK} K and speed < ${LIMITS.heatRpm} rpm`} state={status.heat_dissipation ? "hit" : "ok"} />
          <Row label="Power" value={`${fmt(status.power_w, 0)} W`} rule="normal 3,500 to 9,000 W" state={status.power_out_of_range ? "hit" : "ok"} />
          <Row label="Overstrain" value={`wear × torque ${fmt(status.wear_x_torque, 0)}`} rule="limit 11,000 (L), 12,000 (M), 13,000 (H); tested against the lowest" state={status.possible_overstrain ? "hit" : "ok"} />
          <Row label="Tool wear" value={`${fmt(reading.tool_wear, 0)} min`} rule="failures documented from about 200 min; only a weak signal" state={status.tool_wear_at_failure_range ? "hint" : "ok"} />
        </tbody>
      </table>
      <p className="muted small">Rule check of the latest reading only: no ML, no AI.</p>
    </div>
  );
}
