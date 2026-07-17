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
    activity, confidence = recognizer.predict(dummy_frames)
    
    assert isinstance(activity, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
