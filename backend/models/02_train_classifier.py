"""
Phase 2: ML Failure Classifier for Industrial Root-Cause Investigation Agent

Trains an XGBoost multiclass classifier on sensor readings.

HONEST LIMITATIONS (documented, not hidden):
- "Random Failure" is defined by AI4I2020 as a fixed 0.1% chance INDEPENDENT
  of sensor values. No model can learn to predict it.
- "Tool Wear Failure" triggers at a hidden per-unit random threshold
  (200-240 min) not exposed as a feature, causing heavy overlap (671 normal
  vs 43 failure readings in that same tool_wear range) — expect weak recall.
- Multi-cause failure combinations with FEWER THAN 5 total occurrences are
  bucketed into a single "Multiple Failure Causes (rare combination)" class.
  With only 1-2 examples, a class can't land in both train and test sets, so
  it cannot be meaningfully learned OR evaluated as its own category — kept
  separate anyway, they'd produce a meaningless 0% or 100% recall by chance.
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder

DB_PATH = "industrial_rca.db"
MODEL_OUT = "models/failure_classifier.joblib"
LABEL_MAP_OUT = "models/label_map.json"

FEATURE_COLUMNS = [
    "air_temperature", "process_temperature", "rotational_speed",
    "torque", "tool_wear", "temp_diff", "power"
]

MIN_CLASS_COUNT = 5
RARE_COMBO_LABEL = "Multiple Failure Causes (rare combination)"


# ─────────────────────────────────────────────
# STEP 1: Load data and rebuild labels
# ─────────────────────────────────────────────
def load_labeled_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    readings = pd.read_sql("SELECT * FROM sensor_readings", conn)
    incidents = pd.read_sql("SELECT machine_id, date, failure_type FROM failure_incidents", conn)
    conn.close()

    incidents = incidents.rename(columns={"date": "timestamp"})
    merged = readings.merge(incidents, on=["machine_id", "timestamp"], how="left")
    merged["failure_type"] = merged["failure_type"].fillna("No Failure")

    print("Raw label distribution:")
    print(merged["failure_type"].value_counts())

    # Bucket rare multi-cause combinations (contain " + " and count < MIN_CLASS_COUNT)
    counts = merged["failure_type"].value_counts()
    rare_labels = [
        label for label, count in counts.items()
        if " + " in label and count < MIN_CLASS_COUNT
    ]
    if rare_labels:
        print(f"\nBucketing {len(rare_labels)} rare multi-cause labels into '{RARE_COMBO_LABEL}':")
        for label in rare_labels:
            print(f"  - {label} (n={counts[label]})")
        merged["failure_type"] = merged["failure_type"].replace(rare_labels, RARE_COMBO_LABEL)

    print("\nFinal label distribution (after bucketing):")
    print(merged["failure_type"].value_counts())

    return merged


# ─────────────────────────────────────────────
# STEP 2: Feature engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
    df["power"] = df["torque"] * df["rotational_speed"] * (2 * np.pi / 60)
    return df


# ─────────────────────────────────────────────
# STEP 3: Train
# ─────────────────────────────────────────────
def train():
    df = load_labeled_data()
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y_raw = df["failure_type"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    label_map = {int(i): label for i, label in enumerate(encoder.classes_)}
    print(f"\nClasses ({len(label_map)}): {label_map}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_map),
        eval_metric="mlogloss",
        max_depth=5,
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(
        y_test, y_pred,
        target_names=[label_map[i] for i in sorted(label_map)],
        zero_division=0
    ))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print("Rows = actual, Columns = predicted")
    print("Labels order:", [label_map[i] for i in sorted(label_map)])
    print(cm)

    print("\n=== Feature Importances ===")
    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: -x[1]
    )
    for feat, imp in importances:
        print(f"  {feat}: {imp:.4f}")

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    with open(LABEL_MAP_OUT, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nModel saved to {MODEL_OUT}")
    print(f"Label map saved to {LABEL_MAP_OUT}")

    return model, label_map


if __name__ == "__main__":
    train()