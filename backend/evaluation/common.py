"""
Shared helpers for the Phase 8 evaluation. NO LLM calls in this file.

What it does:
  * re-creates the Phase 2 train/test split EXACTLY (same code, same seed) so we know which incidents
    the classifier never saw,
  * turns failure labels and predictions into sets of catalog causes, and scores them.
"""

import contextlib
import importlib.util
import io
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from graph.checks import (  # noqa: E402
    H_HEAT, H_POWER, H_OVERSTRAIN, H_TOOL_WEAR, compute_limit_checks, required_hypotheses,
)

SENSORS = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
_COMPONENT = {"Heat Dissipation Failure": H_HEAT, "Power Failure": H_POWER,
              "Overstrain Failure": H_OVERSTRAIN, "Tool Wear Failure": H_TOOL_WEAR}


# ---------- labels -> sets of catalog causes ----------

def truth_components(label: str) -> set:
    """The causes a perfect agent should name. Healthy rows, 'Unknown' and 'Random Failure' have no
    sensor-visible cause, so they map to nothing: the right answer there is 'no fault / unexplained'."""
    if label in ("No Failure", "Unknown"):
        return set()
    return {_COMPONENT[p.strip()] for p in label.split(" + ") if p.strip() in _COMPONENT}


def ml_components(pred_class: str) -> set:
    """What the classifier's top class names. 'No Failure', 'Unknown' and the rare-combination bucket
    name no specific cause."""
    if pred_class in ("No Failure", "Unknown", "Multiple Failure Causes (rare combination)"):
        return set()
    return {_COMPONENT[p.strip()] for p in pred_class.split(" + ") if p.strip() in _COMPONENT}


def rule_components(reading: dict, include_tool_wear_signal: bool = True) -> set:
    """Causes forced by the documented rules alone (same function the agent uses), no ML.
    Heat dissipation, power and overstrain are hard limits. Tool wear >= 200 min is only a SIGNAL (about
    94% of readings in that range are healthy), so it can be left out to see the hard rules on their own."""
    checks = compute_limit_checks([reading])
    labels = {label for label, _ in required_hypotheses({"predicted_class": "No Failure"}, checks)}
    if not include_tool_wear_signal:
        labels.discard(H_TOOL_WEAR)
    return labels


def score_case(expected: set, predicted: set) -> str:
    """exact    = named exactly the expected causes (or correctly named nothing when nothing is expected)
       partial  = named some of the expected causes, plus other things or not all of them
       wrong    = named causes, none of them expected
       missed   = expected causes, named nothing
       overclaim= named a cause although the truth has none (healthy, Unknown, Random)"""
    if expected == predicted:
        return "exact"
    if not expected:
        return "overclaim"
    if not predicted:
        return "missed"
    return "partial" if expected & predicted else "wrong"


# ---------- data + split ----------

def load_split():
    """Returns (df, feature_columns). df has one row per sensor reading with:
       in_test (never seen by the classifier), ml_truth (the bucketed label it was trained on),
       raw_label + incident_id (the original failure_incidents label), ml_pred / ml_conf."""
    os.chdir(BACKEND)  # the Phase 2 script uses relative paths
    spec = importlib.util.spec_from_file_location("train02", BACKEND / "models" / "02_train_classifier.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    with contextlib.redirect_stdout(io.StringIO()):
        df = tr.load_labeled_data()
    df = tr.engineer_features(df)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(df["failure_type"])
    _, X_test, _, _ = train_test_split(df[tr.FEATURE_COLUMNS], y, test_size=0.2, random_state=42, stratify=y)
    df["in_test"] = df.index.isin(X_test.index)
    df["ml_truth"] = df["failure_type"]

    conn = sqlite3.connect(tr.DB_PATH)
    lookup = {(m, t): (i, lbl) for i, m, t, lbl in conn.execute(
        "SELECT incident_id, machine_id, date, failure_type FROM failure_incidents")}
    conn.close()
    keys = list(zip(df["machine_id"], df["timestamp"]))
    df["incident_id"] = [lookup.get(k, (None, None))[0] for k in keys]
    df["raw_label"] = [lookup.get(k, (None, "No Failure"))[1] for k in keys]

    import joblib
    model = joblib.load(BACKEND / "models" / "failure_classifier.joblib")
    with open(BACKEND / "models" / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    probs = model.predict_proba(df[tr.FEATURE_COLUMNS])
    df["ml_pred"] = [label_map[i] for i in probs.argmax(axis=1)]
    df["ml_conf"] = probs.max(axis=1)
    return df, tr.FEATURE_COLUMNS


def reading_of(row) -> dict:
    return {c: float(row[c]) for c in SENSORS}
