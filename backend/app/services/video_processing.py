import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class VideoProcessor:
    @staticmethod
    def read_video(video_path: str, max_frames: int = 300) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reads a video file and returns a list of frames along with metadata.
        max_frames: Maximum number of frames to read to prevent memory overload.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")
            
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate sample step if frames exceed maximum
        step = 1
        if total_frames > max_frames:
            step = max(1, total_frames // max_frames)
            
        frames = []
        frame_idx = 0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0:
                frames.append(frame)
                
            frame_idx += 1
            if len(frames) >= max_frames:
                break
                
        cap.release()
        
        duration = float(total_frames) / fps
        metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration if duration > 0 else (len(frames) / fps),
            "total_frames": total_frames
        }
        
        return frames, metadata
