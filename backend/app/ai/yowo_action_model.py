"""Wrapper around the vendored YOWOv2 (github.com/yjh0410/YOWOv2, MIT) repo for
real-time, pretrained human action detection.

Replaces the UCF101/MC3-18 activity label used previously: YOWOv2 does joint
person-detection + action-classification in one pass, pretrained on AVA v2.2
(80 atomic actions - sit, stand, walk, hand wave, hand shake, hug, carry an
object, etc.), which covers everyday small/large actions far more broadly than
UCF101's 7 gym-workout buckets.

The vendored repo (backend/third_party_yowov2/) is added to sys.path lazily,
inside this module only, so its top-level `config`/`models`/`dataset` packages
don't leak into the rest of the app's imports.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .ava_action_met import get_met

_YOWO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party_yowov2")
)
_WEIGHT_PATH = os.path.join(_YOWO_ROOT, "weights", "yowo_v2_tiny_ava.pth")

WINDOW = 16  # frames per inference clip - must match the checkpoint's len_clip
IMG_SIZE = 224
CONF_THRESH = 0.35  # cls_score = sqrt(det_conf * cls_conf) threshold, matches demo.py


class _Args:
    """Mimics the argparse.Namespace YOWOv2's config builders expect."""
    dataset = "ava_v2.2"
    version = "yowo_v2_tiny"
    img_size = IMG_SIZE
    len_clip = WINDOW
    conf_thresh = CONF_THRESH
    nms_thresh = 0.5
    topk = 40
    memory = False
    K = WINDOW


class YowoActionDetector:
    """Loads YOWOv2-Tiny/AVA once; predicts the dominant human action in a
    16-frame window of BGR (OpenCV-style) frames. Falls back to unavailable
    (returns None) if the vendored repo or checkpoint isn't present, so callers
    can degrade gracefully exactly like the other AI wrappers in this app."""

    def __init__(self):
        self.available = False
        self.class_names: List[str] = []
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(_WEIGHT_PATH):
            print(f"YOWOv2 AVA weights not found at {_WEIGHT_PATH}. Action detection disabled.")
            return
        if not os.path.isdir(_YOWO_ROOT):
            print(f"Vendored YOWOv2 repo not found at {_YOWO_ROOT}. Action detection disabled.")
            return

        try:
            if _YOWO_ROOT not in sys.path:
                sys.path.insert(0, _YOWO_ROOT)
            from config import build_dataset_config, build_model_config  # noqa: E402
            from models import build_model  # noqa: E402
            from dataset.transforms import BaseTransform  # noqa: E402

            args = _Args()
            d_cfg = build_dataset_config(args)
            m_cfg = build_model_config(args)
            self.class_names = list(d_cfg["label_map"])

            model, _ = build_model(
                args, d_cfg, m_cfg, self.device, d_cfg["valid_num_classes"], trainable=False
            )
            ckpt = torch.load(_WEIGHT_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            model = model.to(self.device).eval()

            self.model = model
            self.transform = BaseTransform(img_size=IMG_SIZE)
            self.available = True
            print(f"YOWOv2 AVA action detector loaded on {self.device}.")
        except Exception as e:  # noqa: BLE001 - mirror other model loaders' fallback behaviour
            print(f"Failed to load YOWOv2 action detector: {e}. Action detection disabled.")
            self.model = None
            self.available = False

    def _frames_to_clip_tensor(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        if len(frames_bgr) == 1:
            frame_rgb = frames_bgr[0][..., ::-1]
            frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))
            x_list, _ = self.transform([frame_pil])
            x_single = x_list[0]  # [3, H, W]
            x_time = x_single.unsqueeze(1).repeat(1, WINDOW, 1, 1)  # [3, WINDOW, H, W]
            return x_time.unsqueeze(0).to(self.device)
            
        frames_rgb = [f[..., ::-1] for f in frames_bgr]  # BGR -> RGB
        frames_pil = [Image.fromarray(f.astype(np.uint8)) for f in frames_rgb]
        x, _ = self.transform(frames_pil)
        x = torch.stack(x, dim=1).unsqueeze(0)  # [B=1, 3, T, H, W]
        return x.to(self.device)

    def predict(self, frames_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        """All actions above threshold for the most confident detected person,
        as (label, score) pairs sorted by score descending. Empty if no person/
        action clears the confidence threshold, or the model isn't available.

        `frames_bgr` should be the most recent WINDOW frames in temporal order;
        fewer frames are padded by repeating the first frame (same convention
        YOWOv2's own demo.py uses for clip warm-up).
        """
        if not self.available or not frames_bgr:
            return []

        if len(frames_bgr) == 1:
            clip = frames_bgr
        else:
            clip = list(frames_bgr[-WINDOW:])
            if len(clip) < WINDOW:
                clip = [clip[0]] * (WINDOW - len(clip)) + clip

        try:
            x = self._frames_to_clip_tensor(clip)
            with torch.no_grad():
                out = self.model(x)
            bboxes = out[0]  # batch size 1
            if bboxes is None or len(bboxes) == 0:
                return []

            best_det_conf = -1.0
            best_cls_scores = None
            for bbox in bboxes:
                det_conf = float(bbox[4])
                if det_conf > best_det_conf:
                    best_det_conf = det_conf
                    best_cls_scores = bbox[5:]

            if best_cls_scores is None:
                return []

            cls_scores = np.sqrt(max(best_det_conf, 0.0) * np.clip(best_cls_scores, 0.0, None))
            hits = [
                (self.class_names[i], float(cls_scores[i]))
                for i in range(len(cls_scores))
                if cls_scores[i] > CONF_THRESH
            ]
            hits.sort(key=lambda t: t[1], reverse=True)
            return hits
        except Exception as e:  # noqa: BLE001
            print(f"YOWOv2 inference error: {e}")
            return []

    def predict_primary_action(
        self, frames_bgr: List[np.ndarray]
    ) -> Optional[Dict[str, float]]:
        """Convenience wrapper: the single highest-confidence detected action,
        with its hardcoded MET value, or None if nothing cleared threshold."""
        hits = self.predict(frames_bgr)
        if not hits:
            return None
        label, score = hits[0]
        return {"action": label, "confidence": score, "met": get_met(label)}
