"""Investigation jobs: run the agent in the background, record its timeline as it happens, keep the results.

Design choices (all deliberate):
  * ONE investigation at a time (a single worker thread). Others wait in the queue. Groq's free plan allows 8,000
    tokens per minute and 200,000 per day, and an investigation uses about 2,000 to 8,000 tokens.
  * A global hourly cap (DEMO_MAX_PER_HOUR, default 20) so a public demo cannot drain the API key.
  * Results live in memory and are also written to api_runs/<id>.json, so they survive a restart.
"""

import json
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

BACKEND = Path(__file__).resolve().parent.parent
ROUTINE_TEXT = "Routine automated check flagged this machine for investigation."
FAILURE_TEXT = "A failure was reported on this machine. Investigate the root cause."


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def agent_input(req: dict) -> dict:
    """The graph input for a request. The stored incident description is never used (it contains the answer)."""
    reported = req.get("failure_reported", True)
    description = req.get("description") or (FAILURE_TEXT if reported else ROUTINE_TEXT)
    return {"machine_id": req["machine_id"], "incident_description": description, "incident_id": req.get("incident_id"),
            "as_of": req.get("as_of"), "failure_reported": reported, "retry_count": 0}


def classify_error(text: str) -> tuple[str, str]:
    low = text.lower()
    if "tokens per day" in low or "(tpd)" in low:
        return "daily_token_limit", "The demo's daily AI token budget is used up. Please try again later."
    if "429" in low or "rate limit" in low:
        return "rate_limit", "The AI service is rate-limiting requests right now. Please try again in a minute."
    return "internal", "The investigation failed. See the server log for details."


class Budget:
    """At most `max_per_hour` investigations per rolling hour (0 = unlimited)."""

    def __init__(self, max_per_hour: int):
        self.max, self.times, self.lock = max_per_hour, deque(), threading.Lock()

    def allow(self) -> bool:
        if self.max <= 0:
            return True
        with self.lock:
            cutoff = time.time() - 3600
            while self.times and self.times[0] < cutoff:
                self.times.popleft()
            if len(self.times) >= self.max:
                return False
            self.times.append(time.time())
            return True


class RunStore:
    def __init__(self, persist_dir: Optional[Path] = None):
        self._runs: dict[str, dict] = {}
        self._cond = threading.Condition()
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    def create(self, request: dict) -> dict:
        run = {"investigation_id": uuid.uuid4().hex[:12], "status": "queued", "machine_id": request["machine_id"],
               "request": request, "created_at": now_iso(), "started_at": None, "finished_at": None,
               "log": [], "report": None, "error": None, "error_kind": None}
        with self._cond:
            self._runs[run["investigation_id"]] = run
        return run

    def get(self, run_id: str) -> Optional[dict]:
        with self._cond:
            run = self._runs.get(run_id)
            if run is not None:
                return json.loads(json.dumps(run, default=str))     # a copy: callers never touch live state
        if self.persist_dir and (self.persist_dir / f"{run_id}.json").exists():
            try:
                return json.loads((self.persist_dir / f"{run_id}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
        return None

    def list(self, machine_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        with self._cond:
            runs = [json.loads(json.dumps(r, default=str)) for r in self._runs.values()
                    if machine_id is None or r["machine_id"] == machine_id]
        return sorted(runs, key=lambda r: r["created_at"], reverse=True)[:limit]

    def update(self, run_id: str, **fields) -> None:
        with self._cond:
            self._runs[run_id].update(fields)
            self._cond.notify_all()

    def append_log(self, run_id: str, entries: list) -> None:
        if entries:
            with self._cond:
                self._runs[run_id]["log"].extend(entries)
                self._cond.notify_all()

    def finish(self, run_id: str, status: str, report=None, error=None, error_kind=None) -> None:
        with self._cond:                      # log entries are already in; the status flips last
            run = self._runs[run_id]
            run.update(status=status, report=report, error=error, error_kind=error_kind, finished_at=now_iso())
            self._cond.notify_all()
            snapshot = json.loads(json.dumps(run, default=str))
        if self.persist_dir:
            try:
                (self.persist_dir / f"{run_id}.json").write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
            except OSError:
                pass                          # persistence is best-effort; the in-memory copy is authoritative

    def wait(self, run_id: str, after: int, timeout: float = 15.0):
        """Blocks until there are log entries after index `after` or the run is over. Returns (entries, status)."""
        deadline = time.time() + timeout
        with self._cond:
            while True:
                run = self._runs.get(run_id)
                if run is None:
                    return [], "error"
                new = run["log"][after:]
                if new or run["status"] in ("done", "error"):
                    return json.loads(json.dumps(new, default=str)), run["status"]
                remaining = deadline - time.time()
                if remaining <= 0:
                    return [], run["status"]
                self._cond.wait(remaining)


class JobRunner:
    def __init__(self, store: RunStore, graph_factory: Callable, budget: Budget):
        self.store, self.budget = store, budget
        self._factory, self._graph, self._graph_lock = graph_factory, None, threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="investigation")

    def _get_graph(self):
        with self._graph_lock:               # the graph (and its models) is built once, on first use
            if self._graph is None:
                self._graph = self._factory()
            return self._graph

    def submit(self, request: dict) -> dict:
        run = self.store.create(request)
        self._pool.submit(self._run, run["investigation_id"], request)
        return run

    def _run(self, run_id: str, request: dict) -> None:
        self.store.update(run_id, status="running", started_at=now_iso())
        report = None
        try:
            graph = self._get_graph()
            for update in graph.stream(agent_input(request), stream_mode="updates"):
                for _node, payload in update.items():          # one dict per finished node
                    if not isinstance(payload, dict):
                        continue
                    self.store.append_log(run_id, payload.get("investigation_log") or [])
                    if payload.get("final_report"):
                        report = payload["final_report"]
            if report is None:
                raise RuntimeError("the agent finished without producing a report")
            self.store.finish(run_id, "done", report=report)
        except Exception as e:  # noqa: BLE001 - a failed investigation must never kill the worker thread
            kind, friendly = classify_error(f"{type(e).__name__}: {e}")
            print(f"[investigation {run_id}] failed: {type(e).__name__}: {str(e)[:500]}")
            self.store.finish(run_id, "error", error=friendly, error_kind=kind)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def default_graph_factory():
    from graph.build_graph import build_graph     # heavy import (models, vector store): only when first needed
    return build_graph()


def default_runs_dir() -> Path:
    return Path(os.getenv("API_RUNS_DIR", str(BACKEND / "api_runs")))
