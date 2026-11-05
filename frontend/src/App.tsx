import { useEffect, useState } from "react";
import { FleetView } from "./views/FleetView";
import { MachineView } from "./views/MachineView";

function useRoute() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const on = () => setHash(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const m = hash.match(/^#\/machine\/([^/]+)/);
  return m ? ({ name: "machine", id: decodeURIComponent(m[1]) } as const) : ({ name: "fleet" } as const);
}

export default function App() {
  const route = useRoute();
  return (
    <div className="app">
      <header className="topbar">
        <a href="#/" className="brand"><span className="brand-mark" aria-hidden="true" />ROOT-CAUSE MONITOR</a>
        <span className="muted small">Industrial failure investigation · demo</span>
      </header>
      <div className="notice" role="note">
        Demo data: machine IDs, timestamps and maintenance records are <b>synthetic</b>; sensor values and failure labels come from
        the AI4I 2020 dataset. The AI agent explains diagnoses that documented rules and a classifier mostly decide; it is not
        a substitute for an engineer.
      </div>
      <main>{route.name === "machine" ? <MachineView machineId={route.id} /> : <FleetView />}</main>
    </div>
  );
}
