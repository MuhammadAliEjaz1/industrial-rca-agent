"""
Phase 8, step 1: which incidents did the classifier NEVER see, and how optimistic is it on the rest?

Run from backend/:   python evaluation/08_make_eval_sets.py
Writes evaluation/eval_cases.json (cases for the agent run). No LLM, no network.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from evaluation.common import load_split, truth_components, BACKEND

HEALTHY_SAMPLE = 30
HARD_NEGATIVES = 20   # healthy readings with tool wear >= 200 min: the look-alikes of real tool-wear failures
SEED = 0


def main():
    df, _ = load_split()
    train, test = df[~df["in_test"]], df[df["in_test"]]

    print(f"Readings: {len(df)} | train {len(train)} | held-out test {len(test)}")
    acc_tr = (train["ml_pred"] == train["ml_truth"]).mean()
    acc_te = (test["ml_pred"] == test["ml_truth"]).mean()
    print(f"Classifier accuracy: train {acc_tr:.3f} | held-out test {acc_te:.3f}")
    print("  -> If test accuracy is not close to your documented ~0.98, the split was NOT reproduced; tell me.\n")

    print("Recall per class: rows the classifier TRAINED on vs rows it never saw")
    print(f"  {'class':46s} {'n_train':>7s} {'rec_train':>9s} {'n_test':>7s} {'rec_test':>8s}")
    for cls in sorted(df["ml_truth"].unique()):
        a, b = train[train["ml_truth"] == cls], test[test["ml_truth"] == cls]
        ra = f"{(a['ml_pred'] == cls).mean():.2f}" if len(a) else "-"
        rb = f"{(b['ml_pred'] == cls).mean():.2f}" if len(b) else "-"
        print(f"  {cls:46s} {len(a):7d} {ra:>9s} {len(b):7d} {rb:>8s}")

    incidents = df[df["incident_id"].notna()]
    cases = []
    for _, r in incidents.iterrows():
        cases.append({
            "case_id": f"inc-{int(r['incident_id'])}", "kind": "incident", "incident_id": int(r["incident_id"]),
            "machine_id": r["machine_id"], "as_of": r["timestamp"], "truth_raw": r["raw_label"],
            "ml_class_truth": r["ml_truth"], "expected": sorted(truth_components(r["raw_label"])),
            "in_test": bool(r["in_test"]),
        })
    healthy = test[test["raw_label"] == "No Failure"]
    rng = np.random.default_rng(SEED)
    for pos in sorted(rng.choice(len(healthy), size=min(HEALTHY_SAMPLE, len(healthy)), replace=False)):
        r = healthy.iloc[int(pos)]
        cases.append({
            "case_id": f"healthy-{int(r['reading_id'])}", "kind": "healthy", "incident_id": None,
            "machine_id": r["machine_id"], "as_of": r["timestamp"], "truth_raw": "No Failure",
            "ml_class_truth": "No Failure", "expected": [], "in_test": True,
        })

    worn = healthy[healthy["tool_wear"] >= 200]
    for pos in sorted(rng.choice(len(worn), size=min(HARD_NEGATIVES, len(worn)), replace=False)):
        r = worn.iloc[int(pos)]
        if any(c["case_id"] == f"healthy-{int(r['reading_id'])}" for c in cases):
            continue
        cases.append({
            "case_id": f"healthywear-{int(r['reading_id'])}", "kind": "healthy_wear", "incident_id": None,
            "machine_id": r["machine_id"], "as_of": r["timestamp"], "truth_raw": "No Failure",
            "ml_class_truth": "No Failure", "expected": [], "in_test": True,
        })

    out = BACKEND / "evaluation" / "eval_cases.json"
    out.write_text(json.dumps(cases, indent=1), encoding="utf-8")

    held = [c for c in cases if c["kind"] == "incident" and c["in_test"]]
    print(f"\nincidents: {sum(c['kind'] == 'incident' for c in cases)} total, {len(held)} never seen by the classifier")
    print("held-out incidents by true label (these are the per-class sample sizes):")
    counts = {}
    for c in held:
        counts[c["truth_raw"]] = counts.get(c["truth_raw"], 0) + 1
    for lbl, k in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:3d}  {lbl}")
    print(f"healthy held-out cases for false-alarm testing: {sum(c['kind'] == 'healthy' for c in cases)} random "
          f"+ {sum(c['kind'] == 'healthy_wear' for c in cases)} with tool wear >= 200 min (hard negatives)")
    print(f"\nSaved {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
