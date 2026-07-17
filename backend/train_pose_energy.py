"""Train the end-to-end pose -> energy (MET) regressor.

The runtime model (app/ai/pose_energy_model.py) predicts a MET value from a
window of pose frames, and the app converts MET -> calories with
    kcal/min = MET * 3.5 * weight_kg / 200
So the training TARGET is MET, not raw calories.

Three modes
-----------
1) --bootstrap  (works NOW, no dataset needed)
   Trains on procedurally-generated pose windows whose frame-to-frame motion
   magnitude is drawn to span the Compendium-of-Physical-Activities MET range
   (~1 MET at rest to ~12 MET vigorous) - the SAME medical-literature basis
   Vid2Burn uses for its labels. The model learns "normalised motion magnitude
   -> MET" and, because every sample is rendered at a random camera scale/offset
   that normalisation cancels, it also learns to be camera-invariant. This is a
   grounded PROTOTYPE / warm-start, NOT a model trained on real workout video.
   Use it to have a working demo; replace it with mode 3 when you get the data.

       python train_pose_energy.py --bootstrap

2) --smoke
   Tiny synthetic run just to prove the training+save+load pipeline works.

3) real Vid2Burn (once your data application is approved)
   Vid2Burn videos are gated: apply to Kinetics/HMDB/UCF101/NTU-RGBD120, email
   the approvals to the Vid2Burn authors, and you receive a download link. Then:

       python train_pose_energy.py --vid2burn-root /path/to/videos \
                                   --annotation /path/to/calorie_split.pkl

   load_vid2burn() below runs the project's own MediaPipe pose extractor over
   each clip and converts its kcal label to MET - so train and serve use the
   exact same pose representation.

The trained checkpoint is written to models/pose_energy_regressor.pth. As soon
as that file exists, PoseEnergyEstimator.available becomes True and the app uses
it automatically - no code change needed.
"""

import os
import re
import sys
import glob
import pickle
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Make the app package importable when running this file directly.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app.ai.pose_energy_model import (  # noqa: E402
    _PoseEnergyNet,
    PoseEnergyEstimator,
    build_model_input,
    sample_position_window,
    WINDOW,
    NUM_FEATURES,
    NUM_LANDMARKS,
    MODEL_INPUT_DIM,
    _MET_MIN,
    _MET_MAX,
)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "pose_energy_regressor.pth")

Sample = Tuple[List[List[Dict[str, float]]], float]  # (pose_sequence, met)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def calories_to_met(kcal: float, weight_kg: float, minutes: float) -> float:
    """Inverse of kcal = MET * 3.5 * weight / 200 * minutes."""
    if weight_kg <= 0 or minutes <= 0:
        return 0.0
    return kcal * 200.0 / (3.5 * weight_kg * minutes)


def _normalized_frames(pose_sequence: List[List[Dict[str, float]]]) -> List[List[float]]:
    """Normalise every frame the SAME way the runtime does (train/serve parity)."""
    return [
        f for f in (PoseEnergyEstimator._normalize_frame(fr) for fr in pose_sequence)
        if f is not None
    ]


def _frames_to_model_input(pose_sequence: List[List[Dict[str, float]]]) -> np.ndarray:
    """Raw pose sequence -> model input [WINDOW, 198] (position + velocity)."""
    normalized = _normalized_frames(pose_sequence)
    if not normalized:
        return np.zeros((WINDOW, MODEL_INPUT_DIM), dtype=np.float32)
    return build_model_input(normalized)


def _frames_to_position_window(pose_sequence: List[List[Dict[str, float]]]) -> np.ndarray:
    """Raw pose sequence -> [WINDOW, 99] positions (used to measure realised velocity)."""
    normalized = _normalized_frames(pose_sequence)
    if not normalized:
        return np.zeros((WINDOW, NUM_FEATURES), dtype=np.float32)
    return sample_position_window(normalized)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PoseWindowDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        pose_sequence, met = self.samples[i]
        window = _frames_to_model_input(pose_sequence)
        met = float(np.clip(met, _MET_MIN, _MET_MAX))
        return torch.from_numpy(window), torch.tensor(met, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Mode 1: grounded bootstrap (Compendium MET range, camera-invariant)
# --------------------------------------------------------------------------- #
# Approximate MediaPipe standing pose in normalised image space (x right, y down).
_BASE_POSE = {
    0: (0.50, 0.12), 1: (0.48, 0.11), 2: (0.47, 0.11), 3: (0.46, 0.11),
    4: (0.52, 0.11), 5: (0.53, 0.11), 6: (0.54, 0.11), 7: (0.45, 0.12),
    8: (0.55, 0.12), 9: (0.48, 0.15), 10: (0.52, 0.15),
    11: (0.42, 0.28), 12: (0.58, 0.28), 13: (0.38, 0.40), 14: (0.62, 0.40),
    15: (0.36, 0.52), 16: (0.64, 0.52), 17: (0.35, 0.55), 18: (0.65, 0.55),
    19: (0.35, 0.55), 20: (0.65, 0.55), 21: (0.36, 0.54), 22: (0.64, 0.54),
    23: (0.45, 0.55), 24: (0.55, 0.55), 25: (0.44, 0.74), 26: (0.56, 0.74),
    27: (0.44, 0.92), 28: (0.56, 0.92), 29: (0.43, 0.94), 30: (0.57, 0.94),
    31: (0.46, 0.95), 32: (0.54, 0.95),
}
# Limbs that move during exercise (arms + legs); trunk/head mostly bob with body.
_ACTIVE_JOINTS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32]
# Same joints IntensityEngine tracks - used to measure realised motion.
_TRACKED_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]


def _window_velocity(window: np.ndarray) -> float:
    """Mean per-frame displacement of the tracked joints in a [WINDOW, 99] window.

    This is the camera-invariant intensity signal the model must learn to read,
    and it is the SAME quantity a real pose window exposes at inference time.
    """
    total, count = 0.0, 0
    for t in range(1, len(window)):
        for j in _TRACKED_JOINTS:
            a = window[t, 3 * j:3 * j + 3]
            b = window[t - 1, 3 * j:3 * j + 3]
            total += float(np.sqrt(np.sum((a - b) ** 2)))
            count += 1
    return total / count if count else 0.0


def _render_motion_window(rng, u: float) -> List[List[Dict[str, float]]]:
    """Render a raw pose sequence at hidden intensity u, with wide diversity so
    the ONLY reliable predictor of intensity is the realised joint velocity."""
    base = np.array([_BASE_POSE[i] for i in range(NUM_LANDMARKS)], dtype=np.float64)
    limb_amp = 0.015 + u * 0.20
    bob_amp = u * 0.05
    freq = float(rng.uniform(0.5, 3.0))                  # decorrelated from u
    cam_scale = float(rng.uniform(0.7, 1.4))             # normalisation cancels this
    cam_ox = float(rng.uniform(-0.15, 0.15))
    cam_oy = float(rng.uniform(-0.10, 0.10))
    phase = rng.uniform(0, 2 * np.pi, size=NUM_LANDMARKS)
    # A random subset of limbs is active, so no single joint is a shortcut.
    active = [j for j in _ACTIVE_JOINTS if rng.random() < 0.7] or _ACTIVE_JOINTS
    n_frames = int(rng.integers(WINDOW, WINDOW * 3))

    frames: List[List[Dict[str, float]]] = []
    for t in range(n_frames):
        ph = 2 * np.pi * freq * t / max(1, n_frames)
        pts = base.copy()
        for j in active:
            pts[j, 0] += limb_amp * np.sin(ph + phase[j])
            pts[j, 1] += limb_amp * np.cos(ph + phase[j] * 1.3)
        pts[:, 1] += bob_amp * np.sin(ph)
        pts = pts * cam_scale + np.array([cam_ox, cam_oy]) + rng.normal(0, 0.004, size=pts.shape)
        frames.append([
            {"x": float(pts[j, 0]), "y": float(pts[j, 1]),
             "z": float(rng.normal(0, 0.02)), "visibility": 1.0}
            for j in range(NUM_LANDMARKS)
        ])
    return frames


def make_bootstrap_dataset(n: int = 2400, seed: int = 0) -> List[Sample]:
    """Label each window by its REALISED joint velocity mapped onto the
    Compendium MET range, so the model learns the transferable
    'how fast the body moves -> MET' relationship rather than a synthetic quirk."""
    rng = np.random.default_rng(seed)

    # Pass 1: render windows across the full intensity range and measure the
    # realised joint velocity of each.
    rendered = []
    for _ in range(n):
        u = float(rng.uniform(0.0, 1.0))
        frames = _render_motion_window(rng, u)
        vel = _window_velocity(_frames_to_position_window(frames))
        rendered.append((frames, vel))

    # Calibrate an ABSOLUTE velocity->MET scale (median motion ~ 6 MET) so the
    # mapping is dataset-independent and valid on real input.
    vels = np.array([v for _, v in rendered])
    median_v = float(np.median(vels)) or 1e-6
    scale = 5.0 / median_v

    samples: List[Sample] = []
    for frames, vel in rendered:
        met = 1.0 + vel * scale + float(rng.normal(0, 0.4))
        samples.append((frames, float(np.clip(met, _MET_MIN, _MET_MAX))))
    return samples


def make_synthetic_dataset(n: int = 256) -> List[Sample]:
    """Trivial data for --smoke: only proves the pipeline wiring."""
    rng = np.random.default_rng(0)
    samples: List[Sample] = []
    for _ in range(n):
        motion = rng.uniform(0.0, 1.0)
        frames = []
        for t in range(WINDOW + 8):
            base = np.sin(t * motion) * motion
            frames.append([
                {"x": float(base + rng.normal(0, 0.01)),
                 "y": float(base + rng.normal(0, 0.01)),
                 "z": 0.0} for _ in range(NUM_LANDMARKS)
            ])
        samples.append((frames, 1.0 + motion * 10.0))
    return samples


# --------------------------------------------------------------------------- #
# Mode 3: real Vid2Burn adapter (ready to run once you have the videos)
# --------------------------------------------------------------------------- #
def load_vid2burn(vid2burn_root: str, annotation_path: str,
                  max_clips: Optional[int] = None) -> List[Sample]:
    """Build (pose_sequence, met) samples from real Vid2Burn videos.

    Runs the project's own MediaPipe pose extractor over each clip so the pose
    representation matches inference exactly.

    Vid2Burn's annotation is a pickle. Its exact keys vary by release, so adjust
    the three lookups marked TODO to match your file (inspect it with
    `pickle.load` and print one entry). We need per clip: a video path/name, the
    kcal (or kcal/hour) label, and the clip duration to convert kcal -> MET.
    """
    from app.services.video_processing import VideoProcessor
    from app.ai.pose_model import PoseEstimator

    with open(annotation_path, "rb") as f:
        annotation = pickle.load(f)

    # Vid2Burn annotations are typically a list/dict of per-sample entries.
    entries = annotation.values() if isinstance(annotation, dict) else annotation
    pose_estimator = PoseEstimator()
    samples: List[Sample] = []

    for i, entry in enumerate(entries):
        if max_clips is not None and i >= max_clips:
            break
        # ---- TODO: map these to your annotation's actual keys ----------------
        video_name = entry["video_name"]          # TODO
        kcal = float(entry["calorie"])            # TODO (total kcal for the clip)
        duration_min = float(entry["duration"]) / 60.0  # TODO (seconds -> minutes)
        weight_kg = float(entry.get("weight", 70.0))    # reference subject weight
        # ---------------------------------------------------------------------

        matches = glob.glob(os.path.join(vid2burn_root, "**", video_name), recursive=True)
        if not matches:
            continue
        try:
            frames, _meta = VideoProcessor.read_video(matches[0])
            pose_history = pose_estimator.process_video_frames(frames)
            if not pose_history:
                continue
            met = calories_to_met(kcal, weight_kg, duration_min)
            if _MET_MIN <= met <= _MET_MAX:
                samples.append((pose_history, met))
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {video_name}: {e}")

    if not samples:
        raise RuntimeError(
            "No samples built from Vid2Burn. Check --vid2burn-root, --annotation, "
            "and the TODO key mappings in load_vid2burn()."
        )
    print(f"Built {len(samples)} real Vid2Burn samples.")
    return samples


# --------------------------------------------------------------------------- #
# Mode 4: UTD-MHAD adapter (free, ungated - available today)
# --------------------------------------------------------------------------- #
# UTD-MHAD ships no calorie/MET labels, only an action class per clip. MET values
# below are Compendium-of-Physical-Activities-informed estimates for each action
# (same labeling strategy Vid2Burn itself uses) - APPROXIMATE, refine against the
# actual Compendium codes if you need tighter ground truth. Action IDs/names match
# the dataset's published a1..a27 ordering (Chen et al., ICIP 2015).
UTD_MHAD_ACTION_MET: Dict[int, Tuple[str, float]] = {
    1: ("swipe_left", 2.0),
    2: ("swipe_right", 2.0),
    3: ("wave", 2.0),
    4: ("clap", 2.3),
    5: ("throw", 3.0),
    6: ("arm_cross", 2.0),
    7: ("basketball_shoot", 4.0),
    8: ("draw_x", 2.5),
    9: ("draw_circle_cw", 2.5),
    10: ("draw_circle_ccw", 2.5),
    11: ("draw_triangle", 2.5),
    12: ("bowling", 3.5),
    13: ("boxing", 6.0),
    14: ("baseball_swing", 5.0),
    15: ("tennis_swing", 5.0),
    16: ("arm_curl", 3.0),
    17: ("tennis_serve", 6.0),
    18: ("push", 3.0),
    19: ("knock", 2.0),
    20: ("catch", 2.5),
    21: ("pickup_throw", 3.5),
    22: ("jog", 7.0),
    23: ("walk", 3.0),
    24: ("sit2stand", 3.0),
    25: ("stand2sit", 3.0),
    26: ("lunge", 4.0),
    27: ("squat", 5.0),
}

_UTD_FILENAME_RE = re.compile(r"[aA](\d+)_[sS](\d+)_[tT](\d+)", re.IGNORECASE)

# Kinect v1 20-joint order used by UTD-MHAD's d_skel .mat files (empirically verified
# against real clips: joint 0 is HEAD - not hip-center as the raw Microsoft NUI enum
# would suggest - confirmed via a wave clip where only joints 9/10/11 [right
# elbow/wrist/hand] move, and a squat clip where 14/18 [ankles] stay ~static while
# 12/16 [hips] swing ~0.44). Kinect space is metric and Y-UP; MediaPipe is Y-DOWN
# image space, so Y is negated below to match the runtime convention.
_KINECT_HEAD, _KINECT_SHOULDER_CENTER, _KINECT_SPINE, _KINECT_HIP_CENTER = 0, 1, 2, 3
_KINECT_SHOULDER_L, _KINECT_ELBOW_L, _KINECT_WRIST_L, _KINECT_HAND_L = 4, 5, 6, 7
_KINECT_SHOULDER_R, _KINECT_ELBOW_R, _KINECT_WRIST_R, _KINECT_HAND_R = 8, 9, 10, 11
_KINECT_HIP_L, _KINECT_KNEE_L, _KINECT_ANKLE_L, _KINECT_FOOT_L = 12, 13, 14, 15
_KINECT_HIP_R, _KINECT_KNEE_R, _KINECT_ANKLE_R, _KINECT_FOOT_R = 16, 17, 18, 19

# MediaPipe landmark index (0-32) -> source Kinect joint. Every joint that matters to
# PoseEnergyEstimator._normalize_frame (shoulders/hips) and the tracked-velocity set
# (shoulders/elbows/wrists/hips/knees) maps to its exact anatomical Kinect equivalent.
# Face landmarks proxy to HEAD; fingertip/heel landmarks proxy to the nearest real
# joint Kinect actually tracks (hand / ankle-foot) - these aren't used by the model's
# normalization or intensity signal, only fill out the fixed 33-point input shape.
_MEDIAPIPE_FROM_KINECT: Dict[int, int] = {
    0: _KINECT_HEAD, 1: _KINECT_HEAD, 2: _KINECT_HEAD, 3: _KINECT_HEAD,
    4: _KINECT_HEAD, 5: _KINECT_HEAD, 6: _KINECT_HEAD, 7: _KINECT_HEAD,
    8: _KINECT_HEAD, 9: _KINECT_HEAD, 10: _KINECT_HEAD,
    11: _KINECT_SHOULDER_L, 12: _KINECT_SHOULDER_R,
    13: _KINECT_ELBOW_L, 14: _KINECT_ELBOW_R,
    15: _KINECT_WRIST_L, 16: _KINECT_WRIST_R,
    17: _KINECT_HAND_L, 18: _KINECT_HAND_R,
    19: _KINECT_HAND_L, 20: _KINECT_HAND_R,
    21: _KINECT_HAND_L, 22: _KINECT_HAND_R,
    23: _KINECT_HIP_L, 24: _KINECT_HIP_R,
    25: _KINECT_KNEE_L, 26: _KINECT_KNEE_R,
    27: _KINECT_ANKLE_L, 28: _KINECT_ANKLE_R,
    29: _KINECT_ANKLE_L, 30: _KINECT_ANKLE_R,
    31: _KINECT_FOOT_L, 32: _KINECT_FOOT_R,
}


def _kinect_frame_to_mediapipe(frame_xyz: np.ndarray) -> List[Dict[str, float]]:
    """One Kinect frame [20, 3] -> 33 MediaPipe-shaped landmark dicts.

    Y is negated (Kinect is Y-up metric space, MediaPipe is Y-down image space).
    Absolute scale/origin don't matter - _normalize_frame divides by torso length,
    so metric meters vs. normalised image units both wash out identically.
    """
    landmarks = []
    for mp_idx in range(NUM_LANDMARKS):
        src = frame_xyz[_MEDIAPIPE_FROM_KINECT[mp_idx]]
        landmarks.append({
            "x": float(src[0]),
            "y": float(-src[1]),
            "z": float(src[2]),
            "visibility": 1.0,
        })
    return landmarks


def load_utd_mhad_skeleton(
    root: str,
    max_clips: Optional[int] = None,
    actions: Optional[List[int]] = None,
) -> List[Sample]:
    """Build (pose_sequence, met) samples straight from UTD-MHAD's precomputed
    Kinect skeletons - no video decoding or MediaPipe inference needed, so this
    is fast even on CPU. Use when only the (much smaller) Skeleton.zip modality
    is available instead of the RGB videos load_utd_mhad() expects.
    """
    import scipy.io as sio

    mat_paths = sorted(glob.glob(os.path.join(root, "**", "*_skeleton.mat"), recursive=True))
    if not mat_paths:
        raise RuntimeError(f"No *_skeleton.mat files found under {root}")

    samples: List[Sample] = []
    n_used = 0

    for path in mat_paths:
        if max_clips is not None and n_used >= max_clips:
            break

        m = _UTD_FILENAME_RE.search(os.path.basename(path))
        if not m:
            continue
        action_id = int(m.group(1))
        if actions is not None and action_id not in actions:
            continue
        entry = UTD_MHAD_ACTION_MET.get(action_id)
        if entry is None:
            continue
        _action_name, met = entry

        try:
            mat = sio.loadmat(path)
            d_skel = mat["d_skel"]  # [20, 3, n_frames]
            n_frames = d_skel.shape[2]
            pose_history = [
                _kinect_frame_to_mediapipe(d_skel[:, :, t]) for t in range(n_frames)
            ]
            if not pose_history:
                continue
            samples.append((pose_history, met))
            n_used += 1
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {path}: {e}")

    if not samples:
        raise RuntimeError(f"No samples built from UTD-MHAD skeletons under {root}.")
    print(f"Built {len(samples)} real UTD-MHAD skeleton samples from {len(mat_paths)} clips found.")
    return samples


def load_utd_mhad(
    root: str,
    max_clips: Optional[int] = None,
    actions: Optional[List[int]] = None,
) -> List[Sample]:
    """Build (pose_sequence, met) samples from UTD-MHAD RGB clips.

    Runs the project's own MediaPipe pose extractor over each clip (same as
    load_vid2burn) so train and serve share the exact same pose representation.
    MET label comes from UTD_MHAD_ACTION_MET keyed on the action id parsed out
    of the clip filename (aA_sS_tT_color.avi convention), not a per-clip
    ground-truth calorie value - UTD-MHAD has none.

    `actions`: optional whitelist of action ids (e.g. the low/mid-MET arm/hand
    gestures) to focus training on that regime instead of all 27 classes.
    """
    from app.services.video_processing import VideoProcessor
    from app.ai.pose_model import PoseEstimator

    video_paths = sorted(
        glob.glob(os.path.join(root, "**", "*.avi"), recursive=True)
        + glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True)
    )
    if not video_paths:
        raise RuntimeError(f"No .avi/.mp4 clips found under {root}")

    pose_estimator = PoseEstimator()
    samples: List[Sample] = []
    n_used = 0

    for path in video_paths:
        if max_clips is not None and n_used >= max_clips:
            break

        m = _UTD_FILENAME_RE.search(os.path.basename(path))
        if not m:
            continue
        action_id = int(m.group(1))
        if actions is not None and action_id not in actions:
            continue
        entry = UTD_MHAD_ACTION_MET.get(action_id)
        if entry is None:
            continue
        action_name, met = entry

        try:
            frames, _meta = VideoProcessor.read_video(path)
            pose_history = pose_estimator.process_video_frames(frames)
            if not pose_history:
                continue
            samples.append((pose_history, met))
            n_used += 1
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {path}: {e}")

    if not samples:
        raise RuntimeError(
            "No samples built from UTD-MHAD. Check --utd-mhad-root and that "
            "clips follow the aA_sS_tT_*.avi naming convention."
        )
    print(f"Built {len(samples)} real UTD-MHAD samples from {len(video_paths)} clips found.")
    return samples


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(samples: List[Sample], epochs: int, batch_size: int, lr: float, seed: int = 0) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)  # reproducible weight init
    print(f"Training on {device} with {len(samples)} samples.")

    dataset = PoseWindowDataset(samples)
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = _PoseEnergyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # MSE (not Huber) so the model is pushed to fit the full intensity spread
    # instead of collapsing toward the mean MET.
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                val_mae += torch.abs(model(x) - y).sum().item()
        val_mae /= len(val_ds)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_MAE(MET)={val_mae:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"window": WINDOW, "num_features": NUM_FEATURES},
                    "val_mae_met": val_mae,
                },
                CHECKPOINT_PATH,
            )
    print(f"Best val MAE(MET)={best_val:.4f}. Saved -> {CHECKPOINT_PATH}")
    return best_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pose->energy MET regressor.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bootstrap", action="store_true",
                        help="Grounded prototype on Compendium-MET motion priors (works now).")
    parser.add_argument("--smoke", action="store_true", help="Trivial pipeline test.")
    parser.add_argument("--vid2burn-root", type=str, default=None,
                        help="Path to real Vid2Burn video files.")
    parser.add_argument("--annotation", type=str, default=None,
                        help="Path to Vid2Burn calorie annotation pickle.")
    parser.add_argument("--utd-mhad-root", type=str, default=None,
                        help="Path to UTD-MHAD RGB clips (free, ungated dataset).")
    parser.add_argument("--utd-mhad-skeleton-root", type=str, default=None,
                        help="Path to UTD-MHAD precomputed Kinect skeleton .mat clips "
                             "(use this when only the small Skeleton.zip modality is available).")
    parser.add_argument("--utd-mhad-actions", type=str, default=None,
                        help="Comma-separated UTD-MHAD action ids to include (default: all 27).")
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    if args.vid2burn_root and args.annotation:
        samples = load_vid2burn(args.vid2burn_root, args.annotation, args.max_clips)
    elif args.utd_mhad_skeleton_root:
        action_ids = (
            [int(a) for a in args.utd_mhad_actions.split(",")]
            if args.utd_mhad_actions else None
        )
        samples = load_utd_mhad_skeleton(args.utd_mhad_skeleton_root, args.max_clips, action_ids)
    elif args.utd_mhad_root:
        action_ids = (
            [int(a) for a in args.utd_mhad_actions.split(",")]
            if args.utd_mhad_actions else None
        )
        samples = load_utd_mhad(args.utd_mhad_root, args.max_clips, action_ids)
    elif args.bootstrap:
        print("BOOTSTRAP: grounded prototype (Compendium MET priors, camera-invariant).")
        samples = make_bootstrap_dataset()
    elif args.smoke:
        print("SMOKE TEST: trivial synthetic data.")
        samples = make_synthetic_dataset()
        args.epochs = min(args.epochs, 10)
    else:
        parser.error(
            "Choose one: --bootstrap, --smoke, --vid2burn-root + --annotation, "
            "--utd-mhad-root, or --utd-mhad-skeleton-root."
        )

    train(samples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
