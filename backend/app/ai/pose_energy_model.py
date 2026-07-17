"""End-to-end pose -> energy (MET) regressor.

Replaces the old activity -> bucket -> MET-table -> XGBoost calorie chain with a
single Transformer that reads the raw pose sequence and predicts the metabolic
effort directly, keeping the temporal dynamics (tempo, acceleration, which joints
move, range of motion) that the old averaged movement_score threw away.

Design choices that make this robust:
  * The model outputs a MET value (metabolic equivalent), NOT raw calories.
    METs are body-weight independent, so the network only learns the hard part
    (how intense the motion is) and we scale by weight + time with the standard
    ACSM formula that the rest of the codebase already uses:
        kcal/min = MET * 3.5 * weight_kg / 200
    This unifies the live path and the upload path and makes the plain MET table
    a natural fallback.
  * Pose is normalised to be camera-invariant (centred on the hip, scaled by
    torso length) so the estimate no longer depends on how close the person
    stands to the camera - the flaw the old movement_score had.

Until a trained checkpoint (models/pose_energy_regressor.pth) is present, the
estimator reports ``available == False`` and every predict method returns None,
so callers transparently fall back to the existing MET / XGBoost path. This
means dropping the file in nothing changes behaviour; dropping in a trained
checkpoint activates the new model with no code change.
"""

import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# MediaPipe pose landmark indices used for normalisation
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_HIP, _RIGHT_HIP = 23, 24

# Model / input geometry. WINDOW must match what the checkpoint was trained with.
NUM_LANDMARKS = 33
COORDS = 3                      # x, y, z
NUM_FEATURES = NUM_LANDMARKS * COORDS   # 99 position features per frame
WINDOW = 32                     # frames per inference window
VEL_SCALE = 10.0                # amplify velocity channel so motion is salient
MODEL_INPUT_DIM = NUM_FEATURES * 2      # 198: position + velocity per frame

# Physiologically sane MET clamp (rest ~0.9, elite sprint ~20)
_MET_MIN, _MET_MAX = 0.9, 20.0


def sample_position_window(normalized_frames: List[List[float]]) -> np.ndarray:
    """Sample exactly WINDOW frames (linspace) from normalised frames -> [WINDOW, 99]."""
    arr = np.asarray(normalized_frames, dtype=np.float32)
    idx = np.linspace(0, len(arr) - 1, WINDOW).astype(int)
    return arr[idx]


def build_model_input(normalized_frames: List[List[float]]) -> np.ndarray:
    """[WINDOW, 198] = normalised positions concatenated with (scaled) frame-to-frame
    velocity. Exposing velocity explicitly makes motion intensity directly readable,
    which is the transferable, camera-invariant signal for MET. Used by BOTH training
    and inference so the representations match exactly."""
    win = sample_position_window(normalized_frames)          # [WINDOW, 99]
    vel = np.zeros_like(win)
    vel[1:] = win[1:] - win[:-1]
    vel *= VEL_SCALE
    return np.concatenate([win, vel], axis=1)                # [WINDOW, 198]


class _PoseEnergyNet(nn.Module):
    """Small Transformer encoder over a fixed-length pose window -> one MET value."""

    def __init__(
        self,
        n_features: int = MODEL_INPUT_DIM,
        window: int = WINDOW,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window = window
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, 99]
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))     # mean-pool the window
        return self.head(h).squeeze(-1)  # [B] -> MET


class PoseEnergyEstimator:
    """Runtime wrapper around the pose->MET Transformer.

    Loads models/pose_energy_regressor.pth if present. If it is missing or fails
    to load, ``available`` stays False and all predict methods return None so the
    caller can fall back to the MET / XGBoost path.
    """

    def __init__(self):
        models_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models")
        )
        self.model_path = os.path.join(models_dir, "pose_energy_regressor.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[_PoseEnergyNet] = None

        if not os.path.exists(self.model_path):
            print(
                "Pose->energy regressor weights not found at "
                f"{self.model_path}. Falling back to MET/XGBoost calorie path. "
                "Train one with backend/train_pose_energy.py."
            )
            return

        try:
            loaded = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model = _PoseEnergyNet()
            if isinstance(loaded, nn.Module):
                model = loaded
            else:
                state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
                model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            print("Pose->energy regressor loaded successfully.")
        except Exception as e:  # noqa: BLE001 - mirror the other model loaders
            print(f"Failed to load pose->energy regressor: {e}. Using MET/XGBoost fallback.")
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    # ---- pose preprocessing -------------------------------------------------

    @staticmethod
    def _normalize_frame(landmarks: List[Dict[str, float]]) -> Optional[List[float]]:
        """Centre on hip midpoint, scale by torso length -> 99 camera-invariant floats.

        Returns None if the frame does not have the 33 landmarks we need.
        """
        if not landmarks or len(landmarks) < NUM_LANDMARKS:
            return None

        def _mid(a: Dict[str, float], b: Dict[str, float], k: str) -> float:
            return (float(a.get(k, 0.0)) + float(b.get(k, 0.0))) / 2.0

        ls, rs = landmarks[_LEFT_SHOULDER], landmarks[_RIGHT_SHOULDER]
        lh, rh = landmarks[_LEFT_HIP], landmarks[_RIGHT_HIP]

        hip_cx, hip_cy, hip_cz = _mid(lh, rh, "x"), _mid(lh, rh, "y"), _mid(lh, rh, "z")
        sh_cx, sh_cy, sh_cz = _mid(ls, rs, "x"), _mid(ls, rs, "y"), _mid(ls, rs, "z")

        torso = float(np.sqrt((sh_cx - hip_cx) ** 2 + (sh_cy - hip_cy) ** 2 + (sh_cz - hip_cz) ** 2))
        if torso < 1e-6:
            torso = 1.0

        out: List[float] = []
        for lm in landmarks[:NUM_LANDMARKS]:
            out.append((float(lm.get("x", 0.0)) - hip_cx) / torso)
            out.append((float(lm.get("y", 0.0)) - hip_cy) / torso)
            out.append((float(lm.get("z", 0.0)) - hip_cz) / torso)
        return out

    def _model_input_tensor(self, normalized_frames: List[List[float]]) -> torch.Tensor:
        """Normalised frames -> model input tensor [1, WINDOW, 198] (position+velocity)."""
        x = build_model_input(normalized_frames)            # [WINDOW, 198]
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    # ---- prediction ---------------------------------------------------------

    def predict_met(self, pose_window: List[List[Dict[str, float]]]) -> Optional[float]:
        """MET for a window of raw pose frames, or None if unavailable/insufficient."""
        if self.model is None or not pose_window:
            return None

        normalized = [f for f in (self._normalize_frame(fr) for fr in pose_window) if f is not None]
        if not normalized:
            return None

        try:
            x = self._model_input_tensor(normalized)
            with torch.no_grad():
                met = float(self.model(x)[0].item())
            return float(np.clip(met, _MET_MIN, _MET_MAX))
        except Exception as e:  # noqa: BLE001
            print(f"Pose->energy inference error: {e}")
            return None

    def calorie_rate_per_second(
        self, pose_window: List[List[Dict[str, float]]], weight: Optional[float]
    ) -> Optional[float]:
        """kcal/second for the current pose window (for live accumulation)."""
        if not weight:
            return None
        met = self.predict_met(pose_window)
        if met is None:
            return None
        # kcal/min = MET * 3.5 * weight / 200  ->  /60 for per-second
        return met * 3.5 * float(weight) / 200.0 / 60.0

    def predict_calories(
        self,
        pose_history: List[List[Dict[str, float]]],
        weight: Optional[float],
        duration_seconds: float,
        fps: float = 30.0,  # kept for API symmetry; sampling is index-based
    ) -> Optional[float]:
        """Total calories for a full session (upload path), or None if unavailable."""
        if self.model is None or not weight or not pose_history:
            return None

        normalized = [f for f in (self._normalize_frame(fr) for fr in pose_history) if f is not None]
        if not normalized:
            return None

        # Split the session into ~WINDOW-sized chunks so a varying-intensity
        # workout gets a time-aware average MET rather than one global number.
        n_windows = max(1, len(normalized) // WINDOW)
        mets: List[float] = []
        for chunk in np.array_split(np.arange(len(normalized)), n_windows):
            if len(chunk) == 0:
                continue
            frames = [normalized[i] for i in chunk]
            try:
                x = self._model_input_tensor(frames)
                with torch.no_grad():
                    met = float(self.model(x)[0].item())
                mets.append(float(np.clip(met, _MET_MIN, _MET_MAX)))
            except Exception as e:  # noqa: BLE001
                print(f"Pose->energy inference error: {e}")
                continue

        if not mets:
            return None

        mean_met = float(np.mean(mets))
        minutes = max(0.01, duration_seconds / 60.0)
        calories = mean_met * 3.5 * float(weight) / 200.0 * minutes
        return round(max(0.0, calories), 2)
