import { buildChartData } from "../lib/chartData";
import { confidenceTone, splitFact } from "../lib/format";
import type { IncidentRecord, SensorReading } from "../types";

const r = (ts: string, torque: number): SensorReading => ({ timestamp: ts, air_temperature: 298, process_temperature: 308.5, rotational_speed: 1500, torque, tool_wear: 100 });

describe("buildChartData", () => {
  const readings = [r("2025-03-01 08:00:00", 40), r("2025-03-02 08:00:00", 60), r("2025-03-03 08:00:00", 30)];
  const incidents: IncidentRecord[] = [{ incident_id: 9, date: "2025-03-02 08:00:00", failure_type: "Power Failure" }];
  const data = buildChartData(readings, incidents);

  it("numbers readings from 1 and keeps their order", () => {
    expect(data.map((d) => d.i)).toEqual([1, 2, 3]);
  });
  it("derives the temperature gap and the power", () => {
    expect(data[0].gap).toBeCloseTo(10.5, 6);
    expect(data[0].power).toBeCloseTo((40 * 1500 * 2 * Math.PI) / 60, 6); // about 6283 W
  });
  it("marks only the reading that carries an incident", () => {
    expect(data.map((d) => d.incident)).toEqual([undefined, "Power Failure", undefined]);
  });
});

describe("format helpers", () => {
  it("maps confidence levels to tones", () => {
    expect(confidenceTone("HIGH")).toBe("high");
    expect(confidenceTone("medium")).toBe("medium");
    expect(confidenceTone(null)).toBe("none");
    expect(confidenceTone("weird")).toBe("none");
  });
  it("separates computed facts from model-written ones", () => {
    expect(splitFact("[computed] Latest tool wear = 235 min")).toEqual({ computed: true, text: "Latest tool wear = 235 min" });
    expect(splitFact("Technical documentation says X")).toEqual({ computed: false, text: "Technical documentation says X" });
  });
});
