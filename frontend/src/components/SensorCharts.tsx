import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { buildChartData, type ChartPoint } from "../lib/chartData";
import { LIMITS } from "../lib/limits";
import type { IncidentRecord, SensorReading } from "../types";

interface Series { key: keyof ChartPoint; name: string; color: string }
interface Ref { y: number; label: string }

function Panel({ title, unit, data, series, refs }: { title: string; unit: string; data: ChartPoint[]; series: Series[]; refs?: Ref[] }) {
  return (
    <div className="chart-panel">
      <div className="chart-title">
        {title} <span className="muted">{unit}</span>
      </div>
      <ResponsiveContainer width="100%" height={170}>
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#1f2a37" strokeDasharray="3 3" />
          <XAxis dataKey="i" tick={{ fill: "#6b7a8c", fontSize: 10 }} stroke="#2b394a" />
          <YAxis tick={{ fill: "#6b7a8c", fontSize: 10 }} stroke="#2b394a" width={44} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#0e141b", border: "1px solid #2b394a", fontSize: 12 }}
            labelFormatter={(_l, payload) => {
              const p = payload?.[0]?.payload as ChartPoint | undefined;
              return p ? `Reading ${p.i} · ${p.timestamp.slice(0, 19)}${p.incident ? ` · RECORDED FAILURE: ${p.incident}` : ""}` : "";
            }}
          />
          {refs?.map((r) => (
            <ReferenceLine key={r.label} y={r.y} stroke="#f5a524" strokeDasharray="5 4" label={{ value: r.label, fill: "#f5a524", fontSize: 10, position: "insideTopRight" }} />
          ))}
          {data.filter((d) => d.incident).map((d) => (
            <ReferenceLine key={`inc-${d.i}`} x={d.i} stroke="#ff5d5d" strokeOpacity={0.55} />
          ))}
          {series.map((s) => (
            <Line key={s.name} type="monotone" dataKey={s.key as string} name={s.name} stroke={s.color} dot={false} strokeWidth={1.6} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SensorCharts({ readings, incidents }: { readings: SensorReading[]; incidents: IncidentRecord[] }) {
  const data = buildChartData(readings, incidents);
  if (data.length === 0) return <p className="muted">No readings for this machine.</p>;
  return (
    <div>
      <p className="muted small">
        X axis = reading number in the order stored. Red lines mark readings that carry a recorded failure. Timestamps are
        synthetic and the readings are independent samples, so read each chart as a distribution, not a trend.
      </p>
      <div className="chart-grid">
        <Panel title="Temperatures" unit="K" data={data} series={[{ key: "air", name: "Air", color: "#4cc2ff" }, { key: "process", name: "Process", color: "#ff9f5a" }]} />
        <Panel title="Process − air gap" unit="K" data={data} series={[{ key: "gap", name: "Gap", color: "#b794f6" }]} refs={[{ y: LIMITS.heatGapK, label: `${LIMITS.heatGapK} K (heat limit, with speed < ${LIMITS.heatRpm} rpm)` }]} />
        <Panel title="Rotational speed" unit="rpm" data={data} series={[{ key: "rpm", name: "Speed", color: "#5ee0a0" }]} refs={[{ y: LIMITS.heatRpm, label: `${LIMITS.heatRpm} rpm` }]} />
        <Panel title="Torque" unit="Nm" data={data} series={[{ key: "torque", name: "Torque", color: "#f6d365" }]} />
        <Panel title="Derived power" unit="W" data={data} series={[{ key: "power", name: "Power", color: "#ff7a90" }]} refs={[{ y: LIMITS.powerMinW, label: "3,500 W" }, { y: LIMITS.powerMaxW, label: "9,000 W" }]} />
        <Panel title="Tool wear" unit="min" data={data} series={[{ key: "wear", name: "Tool wear", color: "#7dd3c0" }]} refs={[{ y: LIMITS.wearFailMin, label: "200 min (weak signal)" }]} />
      </div>
    </div>
  );
}
