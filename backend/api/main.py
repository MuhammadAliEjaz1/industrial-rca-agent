"""
FastAPI backend. Run from backend/:

    uvicorn api.main:app --reload --port 8000        # interactive docs at http://localhost:8000/docs

Read-only endpoints (no AI, no tokens):   /machines, /machines/{id}, /machines/{id}/sensors, /maintenance, /incidents
Investigations (use the AI agent):        POST /investigations, GET /investigations/{id}, GET /investigations/{id}/events
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import db
from api.models import (
    IncidentHistory, InvestigationAccepted, InvestigationRequest, InvestigationStatus, InvestigationSummary,
    MachineDetail, MachineSummary, MaintenanceHistory, SensorSeries,
)
from api.runs import Budget, JobRunner, RunStore, default_graph_factory, default_runs_dir

TERMINAL = ("done", "error")


def create_app(graph_factory: Optional[Callable] = None, db_path=None, runs_dir=None,
               max_per_hour: Optional[int] = None) -> FastAPI:
    db_path = db_path or db.DEFAULT_DB
    max_per_hour = int(os.getenv("DEMO_MAX_PER_HOUR", "20")) if max_per_hour is None else max_per_hour
    store = RunStore(runs_dir if runs_dir is not None else default_runs_dir())
    runner = JobRunner(store, graph_factory or default_graph_factory, Budget(max_per_hour))

    @asynccontextmanager
    async def lifespan(_app):
        yield
        runner.shutdown()

    app = FastAPI(title="Industrial Root-Cause Investigation Agent API", version="0.1.0", lifespan=lifespan,
                  description="Machines and sensor data (read-only) plus an AI agent that investigates failures and "
                              "writes an evidence-backed report. Machine IDs, timestamps and maintenance records are "
                              "synthetic; see /about.")
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])
    app.state.store, app.state.runner, app.state.db_path = store, runner, db_path

    def need_machine(machine_id: str):
        if not db.machine_exists(machine_id, db_path):
            raise HTTPException(404, f"Unknown machine '{machine_id}'")

    # ---------------- general ----------------
    @app.get("/health", tags=["general"])
    def health():
        return {"status": "ok"}

    @app.get("/about", tags=["general"])
    def about():
        from api.models import DATA_NOTICE
        return {"notices": DATA_NOTICE}

    # ---------------- machines (read-only) ----------------
    @app.get("/machines", response_model=list[MachineSummary], tags=["machines"])
    def machines():
        return db.list_machines(db_path)

    @app.get("/machines/{machine_id}", response_model=MachineDetail, tags=["machines"])
    def machine(machine_id: str):
        m = db.get_machine(machine_id, db_path)
        if m is None:
            raise HTTPException(404, f"Unknown machine '{machine_id}'")
        return m

    @app.get("/machines/{machine_id}/sensors", response_model=SensorSeries, tags=["machines"])
    def sensors(machine_id: str, limit: int = Query(300, ge=1, le=2000),
                as_of: Optional[str] = Query(None, description="Hide readings after this timestamp")):
        need_machine(machine_id)
        rows = db.sensor_series(machine_id, limit, as_of, db_path)
        return {"machine_id": machine_id, "count": len(rows), "readings": rows}

    @app.get("/machines/{machine_id}/maintenance", response_model=MaintenanceHistory, tags=["machines"])
    def maintenance(machine_id: str, as_of: Optional[str] = None):
        need_machine(machine_id)
        rows = db.maintenance(machine_id, as_of, db_path)
        return {"machine_id": machine_id, "count": len(rows), "records": rows}

    @app.get("/machines/{machine_id}/incidents", response_model=IncidentHistory, tags=["machines"])
    def incidents(machine_id: str, as_of: Optional[str] = None):
        need_machine(machine_id)
        rows = db.incidents(machine_id, as_of, db_path)
        return {"machine_id": machine_id, "count": len(rows), "incidents": rows}

    # ---------------- investigations ----------------
    @app.post("/investigations", response_model=InvestigationAccepted, status_code=202, tags=["investigations"])
    def start_investigation(req: InvestigationRequest):
        need_machine(req.machine_id)
        if not runner.budget.allow():
            raise HTTPException(429, "The demo's hourly investigation limit has been reached. Please try again later.")
        run = runner.submit(req.model_dump())
        rid = run["investigation_id"]
        return {"investigation_id": rid, "status": run["status"], "machine_id": req.machine_id,
                "status_url": f"/investigations/{rid}", "events_url": f"/investigations/{rid}/events"}

    @app.get("/investigations", response_model=list[InvestigationSummary], tags=["investigations"])
    def list_investigations(machine_id: Optional[str] = None, limit: int = Query(50, ge=1, le=200)):
        out = []
        for r in store.list(machine_id, limit):
            rep = r.get("report") or {}
            out.append({"investigation_id": r["investigation_id"], "status": r["status"], "machine_id": r["machine_id"],
                        "created_at": r["created_at"], "finished_at": r["finished_at"],
                        "most_supported_cause": rep.get("most_supported_cause"),
                        "confidence_level": rep.get("confidence_level")})
        return out

    @app.get("/investigations/{investigation_id}", response_model=InvestigationStatus, tags=["investigations"])
    def investigation(investigation_id: str):
        run = store.get(investigation_id)
        if run is None:
            raise HTTPException(404, f"Unknown investigation '{investigation_id}'")
        return run

    @app.get("/investigations/{investigation_id}/events", tags=["investigations"],
             description="Server-Sent Events: one 'step' event per finished agent step, then a final 'done' or 'error' "
                         "event. Use ?after=N to resume after N steps already received.")
    def events(investigation_id: str, after: int = Query(0, ge=0)):
        if store.get(investigation_id) is None:
            raise HTTPException(404, f"Unknown investigation '{investigation_id}'")

        def stream():
            index = after
            while True:
                entries, status = store.wait(investigation_id, index, timeout=15.0)
                for e in entries:
                    index += 1
                    yield f"id: {index}\nevent: step\ndata: {json.dumps(e)}\n\n"
                if status in TERMINAL:
                    final = store.get(investigation_id) or {}
                    yield f"event: {status}\ndata: {json.dumps({'status': status, 'error': final.get('error'), 'error_kind': final.get('error_kind')})}\n\n"
                    return
                if not entries:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
