import os
import sys
import numpy as np
import pytest

# Add parent path to sys.path so we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.activity_model import ActivityRecognizer
from app.ai.pose_model import PoseEstimator
from app.ai.intensity import IntensityEngine
from app.ai.calorie_model import CalorieRegressor
from app.ai.yowo_action_model import YowoActionDetector
from app.ai.kinetics_action_model import LifestyleActivityRecognizer
from app.ai.walking_classifier import detect_gait
from app.ai.pose_energy_model import PoseEnergyEstimator

def test_intensity_engine():
    engine = IntensityEngine()
    
    # Create mock landmarks history: 10 frames of 33 landmarks
    history = []
    for t in range(10):
        frame = []
        for idx in range(33):
            # Simulate movements: coordinates shift over time
            val = t * 0.05
            frame.append({"x": val, "y": val, "z": val, "visibility": 0.99})
        history.append(frame)
        
    result = engine.calculate_intensity(history, fps=30.0)
    assert "movement_score" in result
    assert "intensity" in result
    assert result["movement_score"] > 0
    assert result["intensity"] in ["Low", "Medium", "High"]

def test_calorie_regressor():
    regressor = CalorieRegressor()
    
    # Test calorie predictor with metrics
    calories = regressor.predict(
        activity="squat",
        age=25,
        height=175.0,
        weight=70.0,
        duration_seconds=60.0,  # 1 minute
        movement_score=0.5,
        gender="M"
    )
    assert isinstance(calories, float)
    assert calories > 0.0

def test_activity_recognizer():
    recognizer = ActivityRecognizer()

    # Create 16 dummy frames (black images)
    dummy_frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(16)]
    activity, confidence, encoder_class = recognizer.predict(dummy_frames)

    assert isinstance(activity, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert encoder_class in {"Cycling", "HIIT", "Running", "Swimming", "Walking", "Weightlifting", "Yoga"}

def test_yowo_action_detector():
    detector = YowoActionDetector()

    # Create 16 dummy frames (black images) - same shape/size convention as
    # test_activity_recognizer above.
    dummy_frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(16)]
    result = detector.predict_primary_action(dummy_frames)

    # None is a valid outcome (nothing clears the confidence threshold, or the
    # vendored weights aren't available in this environment) - this is a
    # shape/type smoke test, not an accuracy test (see backend/eval_models.py
    # for real accuracy evaluation against labeled clips).
    if result is not None:
        assert isinstance(result, dict)
        assert isinstance(result["action"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["met"], float)

def test_kinetics_lifestyle_recognizer():
    recognizer = LifestyleActivityRecognizer()

    dummy_frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(16)]
    result = recognizer.predict(dummy_frames)

    if result is not None:
        label, confidence = result
        assert isinstance(label, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

def test_walking_classifier():
    # Synthetic alternating-gait pose history: ankles/knees oscillate anti-phase
    # around a realistic shoulder/hip torso scale, so the feature vector has
    # real signal (not all-zero, which correlation can't meaningfully measure).
    history = []
    for t in range(16):
        frame = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99} for _ in range(33)]
        frame[11] = {"x": 0.45, "y": 0.3, "z": 0.0, "visibility": 0.99}  # left shoulder
        frame[12] = {"x": 0.55, "y": 0.3, "z": 0.0, "visibility": 0.99}  # right shoulder
        frame[23] = {"x": 0.45, "y": 0.6, "z": 0.0, "visibility": 0.99}  # left hip
        frame[24] = {"x": 0.55, "y": 0.6, "z": 0.0, "visibility": 0.99}  # right hip

        phase = 0.08 * np.sin(t * 0.8)
        frame[25] = {"x": 0.45, "y": 0.75 + phase * 0.5, "z": 0.0, "visibility": 0.99}  # left knee
        frame[26] = {"x": 0.55, "y": 0.75 - phase * 0.5, "z": 0.0, "visibility": 0.99}  # right knee
        frame[27] = {"x": 0.45, "y": 0.95 + phase, "z": 0.0, "visibility": 0.99}        # left ankle
        frame[28] = {"x": 0.55, "y": 0.95 - phase, "z": 0.0, "visibility": 0.99}        # right ankle
        history.append(frame)

    result = detect_gait(history)
    assert result in (None, "Walking", "Running")

def test_pose_energy_estimator():
    estimator = PoseEnergyEstimator()
    assert isinstance(estimator.available, bool)

    history = []
    for t in range(32):
        frame = [{"x": 0.5 + 0.01 * t, "y": 0.5, "z": 0.0, "visibility": 0.99} for _ in range(33)]
        history.append(frame)

    met = estimator.predict_met(history)
    if met is not None:
        assert isinstance(met, float)
        assert 0.9 <= met <= 20.0

    calories = estimator.predict_calories(pose_history=history, weight=70.0, duration_seconds=60.0, fps=30.0)
    if calories is not None:
        assert isinstance(calories, float)
        assert calories >= 0.0
