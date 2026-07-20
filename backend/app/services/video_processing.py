import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class VideoProcessor:
    # Target effective sampling rate for uploaded videos, independent of the
    # clip's native fps or length. The old approach derived the sample step
    # purely from `total_frames // max_frames`, so the *effective* fps fed to
    # pose estimation/action models varied per upload: a long high-fps video
    # got heavily decimated while a short video kept every native frame
    # regardless of its own fps - an inconsistent input rate across clips of
    # different lengths. Sampling at a fixed target fps instead keeps the
    # temporal density downstream models see consistent.
    TARGET_SAMPLE_FPS = 10.0

    @staticmethod
    def read_video(video_path: str, max_frames: int = 300) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reads a video file and returns a list of frames along with metadata.
        Frames are sampled at a fixed target fps (TARGET_SAMPLE_FPS) rather
        than a frame-count-based stride, so the effective sampling rate is
        consistent regardless of the clip's native fps or length.
        max_frames: hard cap on frames read (prevents memory overload on
        unusually long videos) - no longer what drives the sampling rate.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Stride needed to hit TARGET_SAMPLE_FPS from this clip's native fps
        # (e.g. a 30fps clip samples every 3rd frame to hit ~10fps; a clip
        # already at or below the target keeps every frame).
        step = max(1, round(fps / VideoProcessor.TARGET_SAMPLE_FPS))
        # The fps that the *returned, sub-sampled* frames are actually spaced
        # at - not the native capture fps. Downstream consumers (intensity
        # engine, pose->energy estimator) assume pose_history entries are
        # `1/effective_fps` apart, so reporting the native fps here would
        # silently misinterpret motion speed whenever step > 1.
        effective_fps = fps / step

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
            "fps": effective_fps,
            "width": width,
            "height": height,
            "duration": duration if duration > 0 else (len(frames) / effective_fps),
            "total_frames": total_frames
        }

        return frames, metadata
