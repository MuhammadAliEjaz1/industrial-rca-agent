"""Same file as Phase 2 (backend/models/failure_classifier.py) — reused unmodified in Phase 4."""

import joblib
import json
import numpy as np

MODEL_PATH = "models/failure_classifier.joblib"
LABEL_MAP_PATH = "models/label_map.json"

LOW_RELIABILITY_CLASSES = {
    "Tool Wear Failure",
    "Unknown",
    "Multiple Failure Causes (rare combination)",
}

_model = None
_label_map = None


def _load():
    global _model, _label_map
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(LABEL_MAP_PATH) as f:
            _label_map = {int(k): v for k, v in json.load(f).items()}
    return _model, _label_map


def predict_failure(air_temperature: float, process_temperature: float,
                     rotational_speed: float, torque: float, tool_wear: float) -> dict:
    model, label_map = _load()

    temp_diff = process_temperature - air_temperature
    power = torque * rotational_speed * (2 * np.pi / 60)

    features = np.array([[air_temperature, process_temperature, rotational_speed,
                           torque, tool_wear, temp_diff, power]])

    probs = model.predict_proba(features)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_class = label_map[predicted_idx]

    no_failure_idx = [k for k, v in label_map.items() if v == "No Failure"][0]
    anomaly_score = float(1 - probs[no_failure_idx])
    class_probabilities = {label_map[i]: float(p) for i, p in enumerate(probs)}

    feature_names = ["air_temperature", "process_temperature", "rotational_speed",
                      "torque", "tool_wear", "temp_diff", "power"]
    top_features = sorted(zip(feature_names, model.feature_importances_.tolist()), key=lambda x: -x[1])

    reliability_warning = predicted_class in LOW_RELIABILITY_CLASSES
    reliability_note = None
    if predicted_class == "Tool Wear Failure":
        reliability_note = ("This model has weak historical accuracy (~5% precision, ~12% recall) on "
                             "Tool Wear Failure. Treat as a weak hypothesis requiring evidence verification.")
    elif predicted_class == "Unknown":
        reliability_note = "This class had too few historical examples (9) to be reliably learned or evaluated."
    elif predicted_class == "Multiple Failure Causes (rare combination)":
        reliability_note = ("This class bundles rare multi-cause combinations (7 examples total). "
                             "Investigate contributing sensor readings independently.")

    return {
        "predicted_class": predicted_class,
        "confidence": float(probs[predicted_idx]),
        "anomaly_score": anomaly_score,
        "class_probabilities": class_probabilities,
        "top_features": [{"feature": f, "importance": round(i, 4)} for f, i in top_features],
        "reliability_warning": reliability_warning,
        "reliability_note": reliability_note,
    }
