"""
Phase 8, step 2: the NO-LLM baseline. What do the documented rules and the classifier achieve on their own?
The agent has to beat this to claim the LLM adds value.

Run from backend/:   python evaluation/08_baseline.py
No LLM, no network. Writes evaluation/baseline_results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import (
    load_split, truth_components, ml_components, rule_components, score_case, reading_of, BACKEND,
)

CATS = ["exact", "partial", "wrong", "missed", "overclaim"]
METHODS = {
    "hard rules only": lambda row: rule_components(reading_of(row), include_tool_wear_signal=False),
    "rules + tool-wear hint": lambda row: rule_components(reading_of(row)),
    "classifier only": lambda row: ml_components(row["ml_pred"]),
    "hard rules + classifier": lambda row: rule_components(reading_of(row), False) | ml_components(row["ml_pred"]),
}


def evaluate(rows):
    out = {}
    for name, fn in METHODS.items():
        tally = {c: 0 for c in CATS}
        for _, row in rows.iterrows():
            tally[score_case(truth_components(row["raw_label"]), fn(row))] += 1
        out[name] = tally
    return out


def show(title, rows):
    res = evaluate(rows)
    print(f"\n{title}  (n={len(rows)})")
    print(f"  {'method':24s} " + " ".join(f"{c:>9s}" for c in CATS))
    for name, t in res.items():
        print(f"  {name:24s} " + " ".join(f"{t[c]:9d}" for c in CATS))
    return res


def main():
    df, _ = load_split()
    inc = df[df["incident_id"].notna()]
    held = inc[inc["in_test"]]
    healthy_test = df[(df["raw_label"] == "No Failure") & df["in_test"]]

    results = {
        "held_out_incidents": show("HELD-OUT incidents (classifier never saw them) - the fair test", held),
        "all_incidents": show("ALL incidents (classifier trained on ~80% of these: optimistic for the classifier)", inc),
        "healthy_held_out": show("HEALTHY held-out readings (anything named here is a false alarm)", healthy_test),
    }

    print("\nWhich rule raises false alarms on the healthy held-out readings? (rules + tool-wear hint)")
    flagged = {}
    for _, row in healthy_test.iterrows():
        for label in rule_components(reading_of(row)):  # includes the tool-wear hint
            flagged[label] = flagged.get(label, 0) + 1
    if flagged:
        for label, k in sorted(flagged.items(), key=lambda x: -x[1]):
            print(f"  {k:5d} of {len(healthy_test)}  {label}")
    else:
        print("  none")

    print("\nHeld-out incidents, 'hard rules + classifier', by true label (exact / total):")
    by_label = {}
    for _, row in held.iterrows():
        e = truth_components(row["raw_label"])
        s = score_case(e, rule_components(reading_of(row), False) | ml_components(row["ml_pred"]))
        a = by_label.setdefault(row["raw_label"], [0, 0])
        a[1] += 1
        a[0] += (s == "exact")
    for lbl, (ok, n) in sorted(by_label.items(), key=lambda x: -x[1][1]):
        print(f"  {ok:3d} / {n:<3d}  {lbl}")

    results["held_out_by_label"] = {k: {"exact": v[0], "n": v[1]} for k, v in by_label.items()}
    results["false_alarm_by_rule"] = flagged
    out = BACKEND / "evaluation" / "baseline_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
