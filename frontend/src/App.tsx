import { useEffect, useState } from "react";
import { isStatic } from "./api";
import { FleetView } from "./views/FleetView";
import { MachineView } from "./views/MachineView";
import { RunsView } from "./views/RunsView";
import { parseRoute, type Route } from "./lib/route";

function useRoute(): Route {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const on = () => setHash(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return parseRoute(hash);
}

export default function App() {
  const route = useRoute();
  return (
    <div className="app">
      <header className="topbar">
        <a href="#/" className="brand"><span className="brand-mark" aria-hidden="true" />ROOT-CAUSE MONITOR</a>
        <span className="muted small">Industrial failure investigation · demo</span>
        {isStatic() && <a href="#/runs" className="navlink">Recorded runs</a>}
      </header>
      {isStatic() && (
        <div className="notice notice-static" role="note">
          <b>Recorded demo.</b> There is no live AI or backend on this page. The investigations were run earlier by the real agent
          and are replayed from saved files.
        </div>
      )}
      <div className="notice" role="note">
        Demo data: machine IDs, timestamps and maintenance records are <b>synthetic</b>; sensor values and failure labels come from
        the AI4I 2020 dataset. The AI agent explains diagnoses that documented rules and a classifier mostly decide; it is not
        a substitute for an engineer.
      </div>
      <main>
        {route.name === "machine" ? <MachineView key={`${route.id}/${route.run ?? ""}`} machineId={route.id} openRun={route.run} /> : route.name === "runs" ? <RunsView /> : <FleetView />}
      </main>
    </div>
  );
}
