import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from typing import List, Dict, Optional


class PoseEstimator:
    def __init__(self):
        models_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models")
        )
        self.model_path = os.path.join(models_dir, "pose_landmarker_full.task")
        self.landmarker = None

        if not os.path.exists(self.model_path):
            print(f"Warning: pose_landmarker_full.task not found at {self.model_path}")
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            print("MediaPipe Pose Landmarker (Tasks API) loaded successfully.")
        except Exception as e:
            print(f"Failed to load MediaPipe PoseLandmarker: {e}")

    def process_frame(self, frame: np.ndarray) -> Optional[List[Dict[str, float]]]:
        """
        Process a single BGR frame and return 33 landmark dicts {x, y, z, visibility}.
        Returns None if no pose detected or landmarker not loaded.
        """
        if self.landmarker is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return None

            landmarks = []
            for lm in result.pose_landmarks[0]:
                landmarks.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility) if lm.visibility is not None else 1.0,
                })
            return landmarks
        except Exception as e:
            print(f"Pose frame error: {e}")
            return None

    def process_video_frames(self, frames: List[np.ndarray]) -> List[List[Dict[str, float]]]:
        """Process a list of frames and return pose history."""
        pose_history = []
        for frame in frames:
            landmarks = self.process_frame(frame)
            if landmarks:
                pose_history.append(landmarks)
        return pose_history
