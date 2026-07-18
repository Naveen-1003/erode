"""Wrapper around torchvision's stock Kinetics-400-pretrained MC3-18 for
everyday/lifestyle activities (washing dishes, doing laundry, yoga, ...) that
neither YOWOv2/AVA (80 pose/gesture classes) nor the UCF101 gym-workout
fine-tune in activity_model.py was ever trained to recognize. Kinetics-400 is
a broad, general action-recognition dataset - no custom training or dataset
download needed, torchvision ships the weights and label list directly.

Runs alongside (not instead of) the other two activity models - see
api/prediction.py::_resolve_activity for how the three are combined into one
reported activity name.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.video import mc3_18, MC3_18_Weights

WINDOW = 16  # frames per inference clip, matches activity_model.py's MC3-18 usage
_TOP_K_RUNNER_UP = 5

# Curated everyday/lifestyle activities this model is allowed to name. Out of
# Kinetics-400's 400 classes, most (hundreds of specific sports, musical
# instruments, hobbies, foods, ...) aren't relevant to a fitness/activity
# tracker - same rationale as activity_model.MAIN_WORKOUT_CLASSES and
# ava_action_met.MAIN_ACTIONS. Edit freely to add/remove - every entry must be
# an exact category name from MC3_18_Weights.KINETICS400_V1.meta["categories"].
MAIN_LIFESTYLE_CLASSES = {
    "washing dishes",
    "washing hands",
    "washing feet",
    "washing hair",
    "yoga",
    "mopping floor",
    "sweeping floor",
    "cleaning floor",
    "doing laundry",
    "folding clothes",
    "ironing",
}

# Rough Compendium-of-Physical-Activities-informed MET estimate per class -
# same labeling approach as ava_action_met.AVA_ACTION_MET.
KINETICS_MET = {
    "washing dishes": 2.3,
    "washing hands": 1.8,
    "washing feet": 1.8,
    "washing hair": 2.0,
    "yoga": 3.0,
    "mopping floor": 3.5,
    "sweeping floor": 3.3,
    "cleaning floor": 3.3,
    "doing laundry": 2.3,
    "folding clothes": 2.0,
    "ironing": 2.3,
}


def get_met(label: str, default: float = 2.3) -> float:
    """MET for a curated Kinetics-400 lifestyle label, clamped to a
    physiologically sane range for light household activity."""
    return max(1.5, min(4.0, KINETICS_MET.get(label, default)))


class LifestyleActivityRecognizer:
    """Predicts a curated everyday/lifestyle activity from a short clip, using
    torchvision's off-the-shelf Kinetics-400-pretrained MC3-18. Falls back to
    unavailable (predict() always returns None) if the weights can't be
    downloaded/loaded, so callers degrade gracefully exactly like the other AI
    wrappers in this app."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names: List[str] = []
        self.transforms = None

        try:
            weights = MC3_18_Weights.KINETICS400_V1
            self.model = mc3_18(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            self.class_names = list(weights.meta["categories"])
            self.transforms = weights.transforms()
            print(f"Kinetics-400 MC3-18 lifestyle-activity model loaded on {self.device}.")
        except Exception as e:  # noqa: BLE001 - mirror other model loaders' fallback behaviour
            print(f"Failed to load Kinetics-400 model: {e}. Lifestyle-activity detection disabled.")
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _preprocess(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """BGR frames -> [1, C, T, H, W] via the weights' own official
        transform (resize/crop/normalize exactly as the model was trained)."""
        if len(frames_bgr) == 1:
            clip = list(frames_bgr) * WINDOW
        elif len(frames_bgr) < WINDOW:
            clip = [frames_bgr[0]] * (WINDOW - len(frames_bgr)) + list(frames_bgr)
        else:
            idx = np.linspace(0, len(frames_bgr) - 1, WINDOW).astype(int)
            clip = [frames_bgr[i] for i in idx]

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip]
        clip_np = np.stack(rgb_frames, axis=0)  # [T, H, W, C] uint8
        clip_tensor = torch.from_numpy(clip_np).permute(0, 3, 1, 2)  # [T, C, H, W] uint8
        transformed = self.transforms(clip_tensor)  # -> [C, T, H, W] float, normalized
        return transformed.unsqueeze(0).to(self.device)  # [1, C, T, H, W]

    def predict(self, frames_bgr: List[np.ndarray]) -> Optional[Tuple[str, float]]:
        """(label, confidence) for the highest-confidence curated lifestyle
        class among the model's top guesses, or None if the model isn't
        available, there are no frames, or none of its top candidates is one
        we track (see MAIN_LIFESTYLE_CLASSES)."""
        if not self.available or not frames_bgr:
            return None

        try:
            x = self._preprocess(frames_bgr)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1)[0]

            top_p, top_i = torch.max(probs, dim=0)
            top_class = self.class_names[int(top_i.item())]
            if top_class in MAIN_LIFESTYLE_CLASSES:
                return top_class, float(top_p.item())

            # Best guess isn't one we track - check its next-best runners-up
            # before giving up, same approach as activity_model.py's MC3-18.
            k = min(_TOP_K_RUNNER_UP, len(self.class_names))
            runners_up = torch.topk(probs, k)
            for idx, p in zip(runners_up.indices.tolist(), runners_up.values.tolist()):
                cls = self.class_names[idx]
                if cls in MAIN_LIFESTYLE_CLASSES:
                    return cls, float(p)

            return None
        except Exception as e:  # noqa: BLE001
            print(f"Kinetics-400 inference error: {e}")
            return None
