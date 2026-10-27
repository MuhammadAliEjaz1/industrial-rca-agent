"""Aggregation and printing for the Phase 8 agent results. Pure functions: no database, model or LLM."""

from math import sqrt

CATS = ["exact", "partial", "wrong", "missed", "overclaim"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def group_of(row: dict) -> str:
    return {"incident": "incidents", "healthy_wear": "healthy, tool wear >= 200 min"}.get(row["kind"], "healthy (random)")


def tally(scores) -> dict:
    t = {c: 0 for c in CATS}
    for s in scores:
        t[s] += 1
    return t


def summarize(rows: list, base_scores: dict) -> dict:
    """rows: finished agent rows. base_scores: case_id -> baseline score ('hard rules + classifier')."""
    rows = [r for r in rows if r["case_id"] in base_scores]
    out = {"n": len(rows), "groups": {}, "labels": {}, "regressions": [], "gains": [], "confidence": {},
           "confident_errors": []}
    for g in sorted({group_of(r) for r in rows}):
        sub = [r for r in rows if group_of(r) == g]
        out["groups"][g] = {"n": len(sub), "agent": tally(r["score"] for r in sub),
                            "baseline": tally(base_scores[r["case_id"]] for r in sub)}
    for r in rows:
        if r["kind"] == "incident":
            d = out["labels"].setdefault(r["truth_raw"], {"n": 0, "agent_exact": 0, "base_exact": 0, "named_nothing": 0})
            d["n"] += 1
            d["agent_exact"] += r["score"] == "exact"
            d["base_exact"] += base_scores[r["case_id"]] == "exact"
            d["named_nothing"] += not r["predicted"]
        b = base_scores[r["case_id"]]
        if b == "exact" and r["score"] != "exact":
            out["regressions"].append(r["case_id"])
        if b != "exact" and r["score"] == "exact":
            out["gains"].append(r["case_id"])
        t = out["confidence"].setdefault(r["final_confidence"], {c: 0 for c in CATS})
        t[r["score"]] += 1
        if r["score"] in ("wrong", "overclaim") and r["final_confidence"] in ("high", "medium"):
            out["confident_errors"].append(r["case_id"])
    n = max(len(rows), 1)
    healthy = [r for r in rows if r["kind"] == "healthy"]
    out["faith"] = {
        "unverified_total": sum(r["unverified_facts"] for r in rows),
        "cases_with_unverified": sum(r["unverified_facts"] > 0 for r in rows),
        "capped_candidates": sum(c["capped"] for r in rows for c in r["candidates"]),
        "mean_retries": sum(r["retries"] for r in rows) / n,
        "mean_seconds": sum(r["elapsed_s"] for r in rows) / n,
        "healthy_in_health_mode": (sum(r["mode"] == "health_check" for r in healthy), len(healthy)),
        "history_readings_min_median_max": _spread([r["n_history_readings"] for r in rows if r.get("n_history_readings")]),
    }
    inc = [r for r in rows if r["kind"] == "incident"]
    out["incident_exact_ci"] = (sum(r["score"] == "exact" for r in inc), len(inc),
                                wilson(sum(r["score"] == "exact" for r in inc), len(inc)))
    return out


def _spread(values):
    if not values:
        return None
    v = sorted(values)
    return (v[0], v[len(v) // 2], v[-1])


def format_report(s: dict, n_errors: int = 0) -> str:
    L = [f"Agent results: {s['n']} finished cases" + (f", {n_errors} errored (see the .jsonl)" if n_errors else "")]
    head = f"  {'':26s}" + "".join(f"{c:>10s}" for c in CATS)
    for g, d in s["groups"].items():
        L += ["", f"{g}  (n={d['n']})  - agent vs 'hard rules + classifier' baseline on the SAME cases", head,
              f"  {'agent':26s}" + "".join(f"{d['agent'][c]:10d}" for c in CATS),
              f"  {'baseline':26s}" + "".join(f"{d['baseline'][c]:10d}" for c in CATS)]
    k, n, (lo, hi) = s["incident_exact_ci"]
    if n:
        L += ["", f"Incidents named exactly by the agent: {k}/{n} = {k / n:.0%} (95% interval {lo:.0%} to {hi:.0%})"]
    L += ["", "Incidents by true label  (agent exact / baseline exact / agent named nothing, of n):"]
    for lbl, d in sorted(s["labels"].items(), key=lambda x: -x[1]["n"]):
        L.append(f"  {d['agent_exact']:3d} / {d['base_exact']:3d} / {d['named_nothing']:3d}  of {d['n']:<3d} {lbl}")
    L += ["", f"Regressions (baseline exact, agent not): {len(s['regressions'])}  {s['regressions']}",
          f"Gains (agent exact, baseline not):       {len(s['gains'])}  {s['gains']}"]
    L += ["", "Agent calibration: final confidence vs outcome", f"  {'final confidence':26s}" + "".join(f"{c:>10s}" for c in CATS)]
    for conf in ("high", "medium", "low"):
        if conf in s["confidence"]:
            L.append(f"  {conf:26s}" + "".join(f"{s['confidence'][conf][c]:10d}" for c in CATS))
    L.append(f"  confident errors (wrong or overclaim at medium/high): {len(s['confident_errors'])}  {s['confident_errors']}")
    f = s["faith"]
    ok, tot = f["healthy_in_health_mode"]
    L += ["", "Faithfulness and cost:",
          f"  facts with numbers not found in the evidence: {f['unverified_total']} (in {f['cases_with_unverified']} cases)",
          f"  confidence caps applied: {f['capped_candidates']} | mean retries per case: {f['mean_retries']:.2f}",
          f"  random healthy readings handled in health-check mode: {ok}/{tot}",
          f"  mean seconds per case: {f['mean_seconds']:.0f}",
          f"  history readings visible per case (min / median / max): {f['history_readings_min_median_max']}"]
    return "\n".join(L)
