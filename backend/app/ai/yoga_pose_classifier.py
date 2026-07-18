"""Yoga pose classifier built on the 33 MediaPipe pose landmarks already
computed for every frame (see pose_model.py / intensity.py) - no full video
model, near-zero added cost, which is exactly what live-feed detection needs.

Neither YOWOv2/AVA nor the UCF101/Kinetics-400 action-recognition models have
any concept of individual yoga poses (AVA has no yoga class at all; UCF101
only has generic "TaiChi"; Kinetics-400 has one generic "yoga" catch-all, see
kinetics_action_model.py). Distinguishing *which* pose someone is holding is a
static-geometry problem, not a motion/action-recognition one, so landmark
features on a single frame are the right tool.

Two-tier design:
  1. A real classifier (models/yoga_pose_classifier.pkl), trained by
     train_yoga_pose_classifier.py on ~1000 real labeled yoga photos
     (AtomicLion1/yoga-poses-dataset on Hugging Face) run through this same
     app's MediaPipe pipeline - covers Downward Dog, Tree Pose, Warrior II,
     Goddess Pose, and Plank Pose. This is the accurate path; use it whenever
     the checkpoint is present.
  2. Hand-written geometric rules (_classify_by_rules) as a fallback for when
     the checkpoint is missing, or for poses the dataset doesn't cover at all
     (Child's Pose, Forward Fold) - these were never trained on real data, so
     treat them as a rough approximation, not a tuned model.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

# MediaPipe Pose landmark indices used below
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28

YOGA_POSE_MET = {
    "Tree Pose": 2.5,
    "Warrior II": 3.0,
    "Downward Dog": 3.0,
    "Goddess Pose": 3.2,
    "Plank Pose": 3.5,
    "Child's Pose": 1.5,
    "Forward Fold": 2.5,
}


def get_met(pose: str, default: float = 2.5) -> float:
    return YOGA_POSE_MET.get(pose, default)


def _pt(landmarks: List[Dict[str, float]], idx: int) -> Tuple[float, float]:
    lm = landmarks[idx]
    return lm.get("x", 0.0), lm.get("y", 0.0)


def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle at vertex b (degrees), formed by points a-b-c."""
    a_arr, b_arr, c_arr = np.array(a), np.array(b), np.array(c)
    ba = a_arr - b_arr
    bc = c_arr - b_arr
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) or 1e-6
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# Feature vector shared by training (train_yoga_pose_classifier.py) and
# inference below - order matters, keep both in sync.
FEATURE_NAMES = [
    "l_knee_angle", "r_knee_angle", "l_hip_angle", "r_hip_angle",
    "l_elbow_angle", "r_elbow_angle", "l_shoulder_angle", "r_shoulder_angle",
    "hip_minus_shoulder_y_norm", "ankle_minus_hip_y_norm",
    "wrist_minus_shoulder_y_norm", "wrist_minus_hip_y_norm",
    "ankle_spread_ratio", "wrist_spread_ratio",
]


def extract_pose_features(landmarks: List[Dict[str, float]]) -> Optional[np.ndarray]:
    """Fixed-length, scale/translation-invariant feature vector (joint angles
    + torso-normalized relative positions) from 33 MediaPipe landmarks.
    Scale-invariance (dividing position deltas by torso length) matters
    because it's what lets a classifier trained on dataset photos generalize
    to this app's camera, regardless of how close/far the person stands -
    same normalization idea as pose_energy_model.py. Returns None if
    landmarks are missing/incomplete."""
    if not landmarks or len(landmarks) < 33:
        return None

    l_sh, r_sh = _pt(landmarks, _L_SHOULDER), _pt(landmarks, _R_SHOULDER)
    l_el, r_el = _pt(landmarks, _L_ELBOW), _pt(landmarks, _R_ELBOW)
    l_wr, r_wr = _pt(landmarks, _L_WRIST), _pt(landmarks, _R_WRIST)
    l_hip, r_hip = _pt(landmarks, _L_HIP), _pt(landmarks, _R_HIP)
    l_kn, r_kn = _pt(landmarks, _L_KNEE), _pt(landmarks, _R_KNEE)
    l_an, r_an = _pt(landmarks, _L_ANKLE), _pt(landmarks, _R_ANKLE)

    sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
    hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
    torso = float(np.linalg.norm(np.array(sh_mid) - np.array(hip_mid))) or 1e-6

    l_knee_angle = _angle(l_hip, l_kn, l_an)
    r_knee_angle = _angle(r_hip, r_kn, r_an)
    l_hip_angle = _angle(l_sh, l_hip, l_kn)
    r_hip_angle = _angle(r_sh, r_hip, r_kn)
    l_elbow_angle = _angle(l_sh, l_el, l_wr)
    r_elbow_angle = _angle(r_sh, r_el, r_wr)
    l_shoulder_angle = _angle(l_hip, l_sh, l_el)
    r_shoulder_angle = _angle(r_hip, r_sh, r_el)

    sh_y, hip_y = sh_mid[1], hip_mid[1]
    an_y = (l_an[1] + r_an[1]) / 2.0
    wr_y = (l_wr[1] + r_wr[1]) / 2.0

    an_x_spread = abs(l_an[0] - r_an[0])
    hip_x_spread = abs(l_hip[0] - r_hip[0]) or 1e-6
    wr_x_spread = abs(l_wr[0] - r_wr[0])
    sh_x_spread = abs(l_sh[0] - r_sh[0]) or 1e-6

    return np.array([
        l_knee_angle, r_knee_angle, l_hip_angle, r_hip_angle,
        l_elbow_angle, r_elbow_angle, l_shoulder_angle, r_shoulder_angle,
        (hip_y - sh_y) / torso,
        (an_y - hip_y) / torso,
        (wr_y - sh_y) / torso,
        (wr_y - hip_y) / torso,
        an_x_spread / hip_x_spread,
        wr_x_spread / sh_x_spread,
    ], dtype=np.float32)


# Dataset label -> display name (AtomicLion1/yoga-poses-dataset folder names)
_TRAINED_DISPLAY_NAMES = {
    "downdog": "Downward Dog",
    "tree": "Tree Pose",
    "warrior": "Warrior II",
    "goddess": "Goddess Pose",
    "plank": "Plank Pose",
}

_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "yoga_pose_classifier.pkl")
)
_MIN_TRAINED_CONF = 0.55  # below this, the trained model isn't confident about any of its 5 poses

_trained_model = None
_trained_classes: List[str] = []

if os.path.exists(_MODEL_PATH):
    try:
        with open(_MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        _trained_model = payload["model"]
        _trained_classes = list(payload["classes"])
        print(f"Yoga pose classifier loaded. Classes: {_trained_classes}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load trained yoga pose classifier: {e}. Using geometric rules only.")
else:
    print(f"Trained yoga pose classifier not found at {_MODEL_PATH}. Using geometric-rule fallback only "
          "(covers Downward Dog/Tree Pose/Warrior II/Child's Pose/Forward Fold with hand-tuned thresholds). "
          "Run train_yoga_pose_classifier.py to train the real one.")


def classify_yoga_pose(landmarks: List[Dict[str, float]]) -> Optional[str]:
    """Identify a yoga pose from a single frame's 33 MediaPipe landmarks.
    Prefers the trained classifier (Downward Dog, Tree Pose, Warrior II,
    Goddess Pose, Plank Pose) when available and confident; falls back to
    hand-written geometric rules (Downward Dog, Child's Pose, Tree Pose,
    Warrior II, Forward Fold) otherwise - which also gives Child's Pose and
    Forward Fold a chance even when the trained model is loaded, since it was
    never trained on those two. Returns None if nothing matches confidently -
    a feature meant to fire only on a clear, held pose shouldn't force a
    guess on an ambiguous shape.
    """
    if not landmarks or len(landmarks) < 33:
        return None

    if _trained_model is not None:
        feats = extract_pose_features(landmarks)
        if feats is not None:
            probs = _trained_model.predict_proba([feats])[0]
            top_i = int(np.argmax(probs))
            if probs[top_i] >= _MIN_TRAINED_CONF:
                raw_label = _trained_classes[top_i]
                return _TRAINED_DISPLAY_NAMES.get(raw_label, raw_label)

    return _classify_by_rules(landmarks)


def _classify_by_rules(landmarks: List[Dict[str, float]]) -> Optional[str]:
    """Hand-written geometric fallback - not tuned against real footage, see
    module docstring. Covers Downward Dog, Child's Pose, Tree Pose, Warrior
    II, and Forward Fold."""
    l_sh, r_sh = _pt(landmarks, _L_SHOULDER), _pt(landmarks, _R_SHOULDER)
    l_wr, r_wr = _pt(landmarks, _L_WRIST), _pt(landmarks, _R_WRIST)
    l_hip, r_hip = _pt(landmarks, _L_HIP), _pt(landmarks, _R_HIP)
    l_kn, r_kn = _pt(landmarks, _L_KNEE), _pt(landmarks, _R_KNEE)
    l_an, r_an = _pt(landmarks, _L_ANKLE), _pt(landmarks, _R_ANKLE)

    l_knee_angle = _angle(l_hip, l_kn, l_an)
    r_knee_angle = _angle(r_hip, r_kn, r_an)
    l_hip_angle = _angle(l_sh, l_hip, l_kn)
    r_hip_angle = _angle(r_sh, r_hip, r_kn)

    sh_y = (l_sh[1] + r_sh[1]) / 2.0
    hip_y = (l_hip[1] + r_hip[1]) / 2.0
    an_y = (l_an[1] + r_an[1]) / 2.0
    wr_y = (l_wr[1] + r_wr[1]) / 2.0
    wr_x_spread = abs(l_wr[0] - r_wr[0])
    sh_x_spread = abs(l_sh[0] - r_sh[0]) or 1e-6
    an_x_spread = abs(l_an[0] - r_an[0])
    hip_x_spread = abs(l_hip[0] - r_hip[0]) or 1e-6

    # --- Downward Dog: hips are the highest point of the body (smallest y),
    # legs mostly straight, hands down near the floor/ankles.
    if (
        hip_y < sh_y - 0.05
        and hip_y < an_y - 0.05
        and l_knee_angle > 150
        and r_knee_angle > 150
        and wr_y > sh_y
        and abs(wr_y - an_y) < 0.35
    ):
        return "Downward Dog"

    # --- Child's Pose: kneeling and folded forward - knees sharply bent,
    # torso folded down close to hip/knee height (very low, compact profile).
    if l_knee_angle < 90 and r_knee_angle < 90 and sh_y > hip_y - 0.05 and abs(hip_y - an_y) < 0.25:
        return "Child's Pose"

    # --- Tree Pose: standing tall on one straight leg, the other knee bent
    # with its ankle lifted well above the straight leg's ankle (foot resting
    # against the standing leg rather than on the ground).
    straight_leg_is_left = l_knee_angle > r_knee_angle
    straight_knee = l_knee_angle if straight_leg_is_left else r_knee_angle
    bent_knee = r_knee_angle if straight_leg_is_left else l_knee_angle
    bent_ankle_y = r_an[1] if straight_leg_is_left else l_an[1]
    straight_ankle_y = l_an[1] if straight_leg_is_left else r_an[1]
    if straight_knee > 160 and bent_knee < 100 and (straight_ankle_y - bent_ankle_y) > 0.15:
        return "Tree Pose"

    # --- Warrior II: wide stance, front knee bent ~90 degrees, back leg
    # straight, arms extended out to the sides at roughly shoulder height.
    wide_stance = an_x_spread > hip_x_spread * 2.2
    one_knee_bent_90 = (70 < l_knee_angle < 120) or (70 < r_knee_angle < 120)
    other_knee_straight = l_knee_angle > 150 or r_knee_angle > 150
    arms_out = wr_x_spread > sh_x_spread * 1.8 and abs(wr_y - sh_y) < 0.15
    if wide_stance and one_knee_bent_90 and other_knee_straight and arms_out:
        return "Warrior II"

    # --- Forward Fold: standing with legs straight, torso bent sharply
    # forward at the hips, hands reaching down toward the feet.
    if l_knee_angle > 140 and r_knee_angle > 140 and l_hip_angle < 90 and r_hip_angle < 90 and wr_y > hip_y:
        return "Forward Fold"

    return None
