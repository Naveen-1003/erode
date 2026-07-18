import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional

from .ava_action_met import AVA_ACTION_MET, get_met as get_ava_met
from .yoga_pose_classifier import YOGA_POSE_MET
from .kinetics_action_model import KINETICS_MET

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

def _find_encoder_class(activity: str) -> Optional[str]:
    """Encoder class for this activity label, or None if it doesn't name a
    recognized workout (e.g. it's an everyday action like "sit" or "stand")."""
    act_lower = activity.lower().replace(" ", "").replace("_", "").replace("-", "")
    for key, val in ACTIVITY_MAP.items():
        key_clean = key.lower().replace(" ", "").replace("_", "")
        if key_clean in act_lower or act_lower in key_clean:
            return val
    return None


def map_activity_to_encoder_class(activity: str) -> str:
    """Maps any activity label to one of the 7 encoder classes. Only used to
    satisfy the trained XGBoost model's required input encoding, which has no
    "not a workout" class to fall back to - for MET-based calorie math, use
    `_met_for_activity` instead, which doesn't force everyday actions into a
    workout bucket."""
    return _find_encoder_class(activity) or "HIIT"


# Per-workout MET values for the physiological fallback formulas below.
_MET_MAP: Dict[str, float] = {
    "Running": 9.8,
    "HIIT": 8.0,
    "Weightlifting": 5.0,
    "Cycling": 7.5,
    "Swimming": 8.3,
    "Walking": 3.8,
    "Yoga": 3.0,
}

# Sedentary/light default (sitting, standing, waving, ...) used only when the
# activity label matches neither a workout keyword nor a known AVA action -
# deliberately NOT the HIIT MET, since these actions are not intense exercise.
_REST_MET = 1.5


def _met_for_activity(activity: str) -> float:
    """MET for an activity label. Checks for an exact match against a
    precise, hand-tuned MET table first - AVA everyday actions (e.g. "sit",
    "press", "hand wave"), curated yoga poses ("Tree Pose", ...), and curated
    Kinetics-400 lifestyle activities ("washing dishes", ...) - since a short
    label like "press" can otherwise fuzzy-match a workout keyword (e.g.
    "benchpress") it has nothing to do with. Falls back to the
    workout-keyword search for recognized exercise names (Running, HIIT,
    ...), and finally to a low resting MET for anything unrecognized, rather
    than silently billing it as an intense workout."""
    if activity in AVA_ACTION_MET:
        return get_ava_met(activity)
    if activity in YOGA_POSE_MET:
        return YOGA_POSE_MET[activity]
    if activity in KINETICS_MET:
        return KINETICS_MET[activity]
    matched = _find_encoder_class(activity)
    if matched:
        return _MET_MAP.get(matched, _REST_MET)
    return _REST_MET


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

    def _encode_activity(self, activity: str, encoder_class_hint: Optional[str] = None) -> int:
        """Encode activity string to integer using the fitted LabelEncoder."""
        mapped = encoder_class_hint or map_activity_to_encoder_class(activity)
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

    def _met_fallback(
        self,
        activity: str,
        weight: float,
        duration_mins: float,
        intensity: str = "Medium",
        encoder_class_hint: Optional[str] = None,
    ) -> float:
        """Physiological MET-based fallback when model isn't available."""
        met = _MET_MAP[encoder_class_hint] if encoder_class_hint in _MET_MAP else _met_for_activity(activity)
        intensity_multiplier = self._INTENSITY_MULTIPLIERS.get(intensity, 1.0)
        return met * 3.5 * weight / 200.0 * duration_mins * intensity_multiplier

    def met_rate_per_second(
        self,
        activity: str,
        weight: float,
        intensity: str = "Medium",
        encoder_class_hint: Optional[str] = None,
    ) -> float:
        """Returns calories burned per second for real-time incremental accumulation.

        The XGBoost model is trained on full sessions and returns nonsensical values
        for sub-minute durations. Use this method for live per-frame calorie updates.
        """
        met = _MET_MAP[encoder_class_hint] if encoder_class_hint in _MET_MAP else _met_for_activity(activity)
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
        encoder_class_hint: Optional[str] = None,
        met_hint: Optional[float] = None,
    ) -> float:
        """encoder_class_hint: when the caller already knows the exact workout
        bucket (e.g. from activity_model.ucf_to_encoder_class), pass it here to
        skip fuzzy-matching `activity`'s display text - avoids mismatches like
        "Push Ups" fuzzy-matching a different bucket than the source model
        actually assigned it.

        met_hint: a precise, already-known MET (e.g. a yoga pose or
        Kinetics-400 lifestyle activity with no equivalent in the trained
        model's workout-only vocabulary) - computes calories directly from it
        instead of forcing an unrelated activity through the trained model's
        7-class encoding, which has no way to represent it."""
        duration_mins = max(0.01, duration_seconds / 60.0)

        if met_hint is not None:
            return round(met_hint * 3.5 * weight / 200.0 * duration_mins, 2)

        if self.model is None or self.encoder is None:
            return round(self._met_fallback(activity, weight, duration_mins, encoder_class_hint=encoder_class_hint), 2)

        try:
            activity_encoded = self._encode_activity(activity, encoder_class_hint)

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
            return round(self._met_fallback(activity, weight, duration_mins, encoder_class_hint=encoder_class_hint), 2)
