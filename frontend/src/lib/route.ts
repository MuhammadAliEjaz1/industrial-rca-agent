export type Route =
  | { name: "fleet" }
  | { name: "runs" }
  | { name: "machine"; id: string; run: string | null };

/** '#/' -> fleet, '#/runs' -> recorded runs, '#/machine/M001' or '#/machine/M001/run/inc-105' -> a machine page. */
export function parseRoute(hash: string): Route {
  if (hash === "#/runs") return { name: "runs" };
  const m = hash.match(/^#\/machine\/([^/]+)(?:\/run\/([^/]+))?/);
  if (!m) return { name: "fleet" };
  return { name: "machine", id: decodeURIComponent(m[1]), run: m[2] ? decodeURIComponent(m[2]) : null };
}
