import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict

# The 7 activity classes the encoder was trained on
ENCODER_CLASSES = ['Cycling', 'HIIT', 'Running', 'Swimming', 'Walking', 'Weightlifting', 'Yoga']

# Mapping from UCF101 / MC3-18 / common exercise names → encoder classes
ACTIVITY_MAP = {
    "running": "Running",
    "jogging": "Running",
    "run": "Running",
    "walking": "Walking",
    "walk": "Walking",
    "cycling": "Cycling",
    "biking": "Cycling",
    "bike": "Cycling",
    "riding": "Cycling",
    "swimming": "Swimming",
    "swim": "Swimming",
    "weightlifting": "Weightlifting",
    "lifting": "Weightlifting",
    "deadlift": "Weightlifting",
    "benchpress": "Weightlifting",
    "squat": "Weightlifting",
    "squats": "Weightlifting",
    "pushup": "HIIT",
    "pushups": "HIIT",
    "pullup": "Weightlifting",
    "lunges": "Weightlifting",
    "yoga": "Yoga",
    "hiit": "HIIT",
    "jumping": "HIIT",
    "jumpingjacks": "HIIT",
    "jumping jacks": "HIIT",
    "burpees": "HIIT",
    "burpee": "HIIT",
}

def map_activity_to_encoder_class(activity: str) -> str:
    """Maps any activity label to one of the 7 encoder classes."""
    act_lower = activity.lower().replace(" ", "").replace("_", "").replace("-", "")
    for key, val in ACTIVITY_MAP.items():
        key_clean = key.lower().replace(" ", "").replace("_", "")
        if key_clean in act_lower or act_lower in key_clean:
            return val
    # Default to HIIT (general high-intensity exercise)
    return "HIIT"


class CalorieRegressor:
    def __init__(self):
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        model_path = os.path.join(models_dir, "calorie_model.pkl")
        encoder_path = os.path.join(models_dir, "activity_encoder.pkl")

        self.model = None
        self.encoder = None

        if os.path.exists(encoder_path):
            try:
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                print(f"Activity encoder loaded. Classes: {list(self.encoder.classes_)}")
            except Exception as e:
                print(f"Warning: Could not load activity encoder: {e}")

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"XGBoost calorie model loaded. Features: {list(self.model.feature_names_in_)}")
            except Exception as e:
                print(f"Warning: Could not load calorie model: {e}")

    def _encode_activity(self, activity: str) -> int:
        """Encode activity string to integer using the fitted LabelEncoder."""
        mapped = map_activity_to_encoder_class(activity)
        if self.encoder is not None:
            try:
                return int(self.encoder.transform([mapped])[0])
            except Exception:
                # Fallback: find index manually
                classes = list(self.encoder.classes_)
                if mapped in classes:
                    return classes.index(mapped)
        return 0

    # Discrete multipliers per intensity label.
    # Low=0.2 (near-resting effort), Medium=1.0 (standard MET), High=2.0 (vigorous + EPOC).
    _INTENSITY_MULTIPLIERS = {"Low": 0.2, "Medium": 1.0, "High": 2.0}

    def _met_fallback(self, activity: str, weight: float, duration_mins: float, intensity: str = "Medium") -> float:
        """Physiological MET-based fallback when model isn't available."""
        mapped = map_activity_to_encoder_class(activity)
        met_map = {
            "Running": 9.8,
            "HIIT": 8.0,
            "Weightlifting": 5.0,
            "Cycling": 7.5,
            "Swimming": 8.3,
            "Walking": 3.8,
            "Yoga": 3.0,
        }
        met = met_map.get(mapped, 5.0)
        intensity_multiplier = self._INTENSITY_MULTIPLIERS.get(intensity, 1.0)
        return met * 3.5 * weight / 200.0 * duration_mins * intensity_multiplier

    def met_rate_per_second(self, activity: str, weight: float, intensity: str = "Medium") -> float:
        """Returns calories burned per second for real-time incremental accumulation.

        The XGBoost model is trained on full sessions and returns nonsensical values
        for sub-minute durations. Use this method for live per-frame calorie updates.
        """
        mapped = map_activity_to_encoder_class(activity)
        met_map = {
            "Running": 9.8,
            "HIIT": 8.0,
            "Weightlifting": 5.0,
            "Cycling": 7.5,
            "Swimming": 8.3,
            "Walking": 3.8,
            "Yoga": 3.0,
        }
        met = met_map.get(mapped, 5.0)
        intensity_multiplier = self._INTENSITY_MULTIPLIERS.get(intensity, 1.0)
        calories_per_minute = met * 3.5 * weight / 200.0 * intensity_multiplier
        return calories_per_minute / 60.0

    def predict(
        self,
        activity: str,
        age: int,
        height: float,
        weight: float,
        duration_seconds: float,
        movement_score: float,
        gender: str = "M",  # kept for API compatibility but not used by model
    ) -> float:
        duration_mins = max(0.01, duration_seconds / 60.0)

        if self.model is None or self.encoder is None:
            return round(self._met_fallback(activity, weight, duration_mins, movement_score), 2)

        try:
            activity_encoded = self._encode_activity(activity)

            # Build DataFrame with exact feature names the model expects
            data = pd.DataFrame([{
                "activity_encoded": activity_encoded,
                "age": age,
                "height": height,
                "weight": weight,
                "duration": duration_mins,
                "movement_score": movement_score,
            }])

            pred = float(self.model.predict(data)[0])
            return round(max(0.0, pred), 2)

        except Exception as e:
            print(f"Calorie model prediction error: {e}. Using MET fallback.")
            return round(self._met_fallback(activity, weight, duration_mins, movement_score), 2)
