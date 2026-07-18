"""Walking detector on the pose-landmark history already tracked for
intensity scoring (see intensity.py). Neither UCF101 (MC3-18) nor
Kinetics-400 has a plain "walking" class (the only options are
dog/horse/treadmill-qualified variants), so this exists to catch plain
walking without relying on those mismatched proxies.

Walking is fundamentally a MOTION question (a repeating gait cycle), not a
single-frame pose question like the yoga poses in yoga_pose_classifier.py:
one foot lifts and swings while the other stays grounded, and they
alternate every stride. The feature vector below captures that: each
ankle's (and knee's) vertical lift relative to the hip, scaled by torso
length (shoulder-to-hip distance - a stable reference that barely moves
during a stride, unlike ankle-to-hip distance, which would be
self-referential if used as the scale for the ankle-lift signal itself),
plus how strongly the two legs' lift alternates (anti-phase correlation)
and how much the hips translate sideways (rules out in-place arm/torso
motion that isn't actually gait).

Two-tier design, same pattern as yoga_pose_classifier.py:
  1. A real classifier (models/walking_classifier.pkl), trained by
     train_walking_classifier.py on real HMDB51 "walk" clips (positive) vs.
     ten other action classes (negative) - see that script for the full
     class list and methodology. Use it whenever the checkpoint is present.
  2. A hand-written correlation-threshold rule as a fallback for when the
     checkpoint is missing - never validated against real footage, treat it
     as a rough approximation only.

Caveats:
  - HMDB51 is real but noisy "in the wild" movie footage (camera motion,
    cuts, occlusion) - a much harder source than the yoga classifier's clean
    posed photos, and gait is a temporal/motion signal, inherently noisier to
    extract than a static pose. Measured held-out performance (see
    train_walking_classifier.py's output) is genuinely mediocre: at the
    default decision threshold "walk" precision is only ~51%. This is why
    _MIN_TRAINED_CONF below is set high (favoring precision over recall) and
    why this stays a last-resort fallback rather than a primary detector -
    AVA's own "walk" class (real supervised training data, well-represented
    as one of its 14 core "Pose" categories) remains the trustworthy path.
  - HMDB51 clips are real video at native (~25-30fps) frame rates, so the
    trained classifier is calibrated for a properly-sampled gait signal. The
    live feed only captures at 1 FPS - a full stride cycle is roughly 1-2 Hz,
    so 1 FPS is undersampling the very thing being detected. Expect this to
    work better against the video-upload path (real video FPS) than live.
"""

import os
import pickle
from typing import Dict, List, Optional

import numpy as np

# MediaPipe Pose landmark indices (33-point format, used by this app's own
# pose pipeline - see pose_model.py)
_MP_L_SHOULDER, _MP_R_SHOULDER = 11, 12
_MP_L_HIP, _MP_R_HIP = 23, 24
_MP_L_KNEE, _MP_R_KNEE = 25, 26
_MP_L_ANKLE, _MP_R_ANKLE = 27, 28

WINDOW = 16              # matches the ~16-frame windows used elsewhere in this app
MIN_FRAMES = 6           # fewer than this and there's no real cycle to measure

FEATURE_NAMES = [
    "l_ankle_lift_std", "r_ankle_lift_std", "ankle_lift_corr",
    "l_knee_lift_std", "r_knee_lift_std", "knee_lift_corr",
    "hip_x_std",
]


def _simple_frame_from_mediapipe(landmarks: List[Dict[str, float]]) -> Optional[dict]:
    """Adapter: this app's 33-point MediaPipe landmark dicts -> the plain
    per-frame joint dict extract_gait_features() operates on."""
    if not landmarks or len(landmarks) < 33:
        return None
    l_sh, r_sh = landmarks[_MP_L_SHOULDER], landmarks[_MP_R_SHOULDER]
    l_hip, r_hip = landmarks[_MP_L_HIP], landmarks[_MP_R_HIP]
    l_kn, r_kn = landmarks[_MP_L_KNEE], landmarks[_MP_R_KNEE]
    l_an, r_an = landmarks[_MP_L_ANKLE], landmarks[_MP_R_ANKLE]
    return _make_simple_frame(
        (l_sh.get("x", 0.0), l_sh.get("y", 0.0)), (r_sh.get("x", 0.0), r_sh.get("y", 0.0)),
        (l_hip.get("x", 0.0), l_hip.get("y", 0.0)), (r_hip.get("x", 0.0), r_hip.get("y", 0.0)),
        (l_kn.get("x", 0.0), l_kn.get("y", 0.0)), (r_kn.get("x", 0.0), r_kn.get("y", 0.0)),
        (l_an.get("x", 0.0), l_an.get("y", 0.0)), (r_an.get("x", 0.0), r_an.get("y", 0.0)),
    )


def _make_simple_frame(l_sh, r_sh, l_hip, r_hip, l_kn, r_kn, l_an, r_an) -> dict:
    """Reduce raw joint (x, y) pairs (any coordinate system - normalized
    MediaPipe or raw-pixel COCO, doesn't matter since everything below is a
    ratio) to the handful of scalars extract_gait_features() needs."""
    hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
    sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
    # Full 2D distance, not just the y-difference: for a person lying down,
    # crouched, or at an unusual camera angle, shoulder and hip can land at
    # nearly the same image height even though they're clearly apart in the
    # frame - using y-only would let this collapse toward zero and blow up
    # every ratio below. max(..., small floor) guards near-zero, not just
    # exact zero (an `x or 1e-6` idiom wouldn't catch a merely-tiny value).
    torso = max(float(np.hypot(sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])), 1e-3)
    return {
        "hip_mid_x_norm": hip_mid[0] / torso,
        "l_ankle_lift": -(l_an[1] - hip_mid[1]) / torso,
        "r_ankle_lift": -(r_an[1] - hip_mid[1]) / torso,
        "l_knee_lift": -(l_kn[1] - hip_mid[1]) / torso,
        "r_knee_lift": -(r_kn[1] - hip_mid[1]) / torso,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_gait_features(simple_frames: List[dict]) -> Optional[np.ndarray]:
    """Fixed-length feature vector (order matches FEATURE_NAMES) from a
    sequence of _make_simple_frame() dicts - shared by training
    (train_walking_classifier.py) and inference (is_walking() below)."""
    if len(simple_frames) < MIN_FRAMES:
        return None

    l_ankle = np.array([f["l_ankle_lift"] for f in simple_frames])
    r_ankle = np.array([f["r_ankle_lift"] for f in simple_frames])
    l_knee = np.array([f["l_knee_lift"] for f in simple_frames])
    r_knee = np.array([f["r_knee_lift"] for f in simple_frames])
    hip_x = np.array([f["hip_mid_x_norm"] for f in simple_frames])

    return np.array([
        l_ankle.std(), r_ankle.std(), _safe_corr(l_ankle, r_ankle),
        l_knee.std(), r_knee.std(), _safe_corr(l_knee, r_knee),
        hip_x.std(),
    ], dtype=np.float32)


_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "walking_classifier.pkl")
)
_MIN_TRAINED_CONF = 0.78  # P(walk) must clear this - see train_walking_classifier.py's precision/recall
                          # trade-off: at 0.5 the model is only ~51% precise ("walk" is genuinely harder to
                          # learn from noisy in-the-wild HMDB51 video than the yoga poses were from clean
                          # photos); 0.78 trades recall (~26%) for ~72% precision, appropriate since this is
                          # a last-resort fallback - better to stay quiet than confidently mislabel something.

_trained_model = None

if os.path.exists(_MODEL_PATH):
    try:
        with open(_MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        _trained_model = payload["model"]
        print("Walking classifier loaded (trained on HMDB51 gait data).")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load trained walking classifier: {e}. Using geometric-rule fallback only.")
else:
    print(f"Trained walking classifier not found at {_MODEL_PATH}. Using geometric-rule fallback only "
          "(untuned anti-phase correlation threshold). Run train_walking_classifier.py to train the real one.")


def is_walking(pose_window: List[List[Dict[str, float]]]) -> bool:
    """True if the recent pose history shows a walking-like gait."""
    if not pose_window or len(pose_window) < MIN_FRAMES:
        return False

    window = pose_window[-WINDOW:]
    simple_frames = [f for f in (_simple_frame_from_mediapipe(lm) for lm in window) if f is not None]
    feats = extract_gait_features(simple_frames)
    if feats is None:
        return False

    if _trained_model is not None:
        prob = float(_trained_model.predict_proba([feats])[0][1])
        return prob >= _MIN_TRAINED_CONF

    return _classify_by_rule(feats)


def _classify_by_rule(feats: np.ndarray) -> bool:
    """Hand-written fallback - not tuned against real footage, see module
    docstring. feats order matches FEATURE_NAMES."""
    l_ankle_std, r_ankle_std, ankle_corr, _, _, _, _ = feats
    min_swing_std = 0.03
    max_anti_phase_corr = -0.3
    if l_ankle_std < min_swing_std or r_ankle_std < min_swing_std:
        return False
    return ankle_corr < max_anti_phase_corr
