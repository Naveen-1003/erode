import numpy as np
from typing import List, Dict, Any

class IntensityEngine:
    # MediaPipe Landmark Indices:
    # Left Shoulder: 11, Right Shoulder: 12
    # Left Wrist: 15, Right Wrist: 16
    # Left Hip: 23, Right Hip: 24
    # Left Knee: 25, Right Knee: 26
    TRACKED_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]

    @staticmethod
    def calculate_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
        return np.sqrt(
            (p1.get('x', 0) - p2.get('x', 0))**2 +
            (p1.get('y', 0) - p2.get('y', 0))**2 +
            (p1.get('z', 0) - p2.get('z', 0))**2
        )

    def calculate_intensity(self, pose_history: List[List[Dict[str, float]]], fps: float = 30.0) -> Dict[str, Any]:
        """
        Calculates joint velocity over a sequence of poses.
        pose_history: List of frames, where each frame is a list of 33 landmark dicts {'x', 'y', 'z'}
        fps: Frame rate of the video sequence to compute actual velocity.
        """
        if len(pose_history) < 2:
            return {"movement_score": 0.0, "intensity": "Low"}

        dt = 1.0 / fps
        velocities = []

        for t in range(1, len(pose_history)):
            prev_frame = pose_history[t - 1]
            curr_frame = pose_history[t]

            # If the frame data is empty or incomplete, skip
            if not prev_frame or not curr_frame or len(prev_frame) < 33 or len(curr_frame) < 33:
                continue

            frame_velocities = []
            for idx in self.TRACKED_LANDMARKS:
                p1 = prev_frame[idx]
                p2 = curr_frame[idx]
                
                # Calculate Euclidean distance
                dist = self.calculate_distance(p1, p2)
                # velocity = distance / time
                vel = dist / dt
                frame_velocities.append(vel)
            
            if frame_velocities:
                velocities.append(np.mean(frame_velocities))

        if not velocities:
            return {"movement_score": 0.0, "intensity": "Low"}

        # Calculate average velocity (movement score) across all frames
        movement_score = float(np.mean(velocities))

        # Thresholds are calibrated at 10 FPS. Scale down linearly for lower frame
        # rates so that 1 FPS live streaming classifies intensity correctly.
        fps_scale = fps / 10.0
        if movement_score < 0.3 * fps_scale:
            intensity = "Low"
        elif movement_score < 0.7 * fps_scale:
            intensity = "Medium"
        else:
            intensity = "High"

        return {
            "movement_score": round(movement_score, 4),
            "intensity": intensity
        }
