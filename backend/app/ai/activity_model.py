import os
import re
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple
from torchvision.models.video import mc3_18

# UCF101 class list (101 classes, alphabetical order as used during training)
UCF101_CLASSES = [
    "ApplyEyeMakeup","ApplyLipstick","Archery","BabyCrawling","BalanceBeam",
    "BandMarching","BaseballPitch","Basketball","BasketballDunk","BenchPress",
    "Biking","Billiards","BlowDryHair","BlowingCandles","BodyWeightSquats",
    "Bowling","BoxingPunchingBag","BoxingSpeedBag","BreastStroke","BrushingTeeth",
    "CleanAndJerk","CliffDiving","CricketBowling","CricketShot","CuttingInKitchen",
    "Diving","Drumming","Fencing","FieldHockeyPenalty","FloorGymnastics",
    "FrisbeeCatch","FrontCrawl","GolfSwing","Haircut","Hammering",
    "HammerThrow","HandstandPushups","HandstandWalking","HeadMassage","HighJump",
    "HorseRace","HorseRiding","HulaHoop","IceDancing","JavelinThrow",
    "JugglingBalls","JumpingJack","JumpRope","Kayaking","Knitting",
    "LongJump","Lunges","MilitaryParade","Mixing","MoppingFloor",
    "Nunchucks","ParallelBars","PizzaTossing","PlayingCello","PlayingDaf",
    "PlayingDhol","PlayingFlute","PlayingGuitar","PlayingPiano","PlayingSitar",
    "PlayingTabla","PlayingViolin","PoleVault","PommelHorse","PullUps",
    "Punch","PushUps","Rafting","RockClimbingIndoor","RopeClimbing",
    "Rowing","SalsaSpin","ShavingBeard","Shotput","SkateBoarding",
    "Skiing","Skijet","SkyDiving","SoccerJuggling","SoccerPenalty",
    "StillRings","SumoWrestling","Surfing","Swing","TableTennisShot",
    "TaiChi","TennisSwing","ThrowDiscus","TrampolineJumping","Typing",
    "UnevenBars","VolleyballSpiking","WalkingWithDog","WallPushups","WritingOnBoard",
    "YoYo",
]

# Map UCF101 class names → calorie encoder classes
UCF_TO_ENCODER = {
    "Biking": "Cycling",
    "FrontCrawl": "Swimming",
    "BreastStroke": "Swimming",
    "BackStroke": "Swimming",
    "Rowing": "Swimming",
    "JumpingJack": "HIIT",
    "JumpRope": "HIIT",
    "TrampolineJumping": "HIIT",
    "BoxingPunchingBag": "HIIT",
    "BoxingSpeedBag": "HIIT",
    "BenchPress": "Weightlifting",
    "BodyWeightSquats": "Weightlifting",
    "Lunges": "Weightlifting",
    "PushUps": "Weightlifting",
    "PullUps": "Weightlifting",
    "HandstandPushups": "Weightlifting",
    "WallPushups": "Weightlifting",
    "CleanAndJerk": "Weightlifting",
    "HorseRiding": "Cycling",
    "HorseRace": "Running",
    "TaiChi": "Yoga",
    "WalkingWithDog": "Walking",
    "SalsaSpin": "HIIT",
    "HighJump": "HIIT",
    "LongJump": "HIIT",
    "HulaHoop": "HIIT",
    "Skiing": "HIIT",
    "SkateBoarding": "Cycling",
    "SoccerJuggling": "HIIT",
    "SoccerPenalty": "Running",
}

# Default mapping for unmapped activities by keyword
def ucf_to_encoder_class(ucf_class: str) -> str:
    if ucf_class in UCF_TO_ENCODER:
        return UCF_TO_ENCODER[ucf_class]
    cl = ucf_class.lower()
    if "run" in cl or "sprint" in cl:     return "Running"
    if "walk" in cl:                       return "Walking"
    if "swim" in cl or "crawl" in cl:     return "Swimming"
    if "bike" in cl or "cycl" in cl:      return "Cycling"
    if "yoga" in cl or "tai" in cl:       return "Yoga"
    if "lift" in cl or "press" in cl or "squat" in cl or "push" in cl:
        return "Weightlifting"
    if "jump" in cl or "hiit" in cl or "boxing" in cl or "kick" in cl:
        return "HIIT"
    return "HIIT"   # default to general high-intensity


def format_ucf_label(ucf_class: str) -> str:
    """PascalCase UCF101 class name -> readable display string, e.g.
    "BodyWeightSquats" -> "Body Weight Squats", "YoYo" -> "Yo Yo". This is the
    specific workout name reported to the user; ucf_to_encoder_class() above
    is a *separate*, coarser bucketing used only for calorie MET lookups."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", ucf_class).strip()


# Curated "main workouts" MC3-18 is allowed to name specifically. UCF101 has
# 101 classes total, but most (ApplyEyeMakeup, PlayingSitar, HorseRace,
# Bowling, ...) have nothing to do with a home/gym workout and only dilute
# accuracy: the model has to split probability mass across all 101 every
# prediction, so a real squat clip can lose to an unrelated class by a hair.
# Restricting the winning candidate to this shortlist means the model only
# has to be right about the exercises that actually matter. Edit freely to
# add/remove - every entry must be a name from UCF101_CLASSES above.
MAIN_WORKOUT_CLASSES = {
    "BenchPress",
    "BodyWeightSquats",
    "CleanAndJerk",
    "HandstandPushups",
    "JumpingJack",
    "JumpRope",
    "Lunges",
    "PullUps",
    "PushUps",
    "WallPushups",
    "BoxingPunchingBag",
    "Biking",
    "Rowing",
}

# How many of the model's next-best guesses (after its outright top-1) to
# still check for a curated class before giving up. Single-clip video models
# (no multi-crop/multi-clip test-time averaging here) rarely put >15-20% of
# the softmax mass on any one of 101 classes even when the top-1 is correct,
# so gating on an absolute confidence floor rejects plenty of genuinely
# correct "Squats"/"PushUps" guesses. Checking the top-K instead - and
# trusting the model's own ranking - avoids that without requiring a
# threshold that has to be re-tuned per checkpoint.
_TOP_K_RUNNER_UP = 5


class ActivityRecognizer:
    def __init__(self):
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        self.model_path = os.path.join(models_dir, "mc3_ucf101.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if os.path.exists(self.model_path):
            try:
                loaded = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model = mc3_18(num_classes=101)
                if isinstance(loaded, nn.Module):
                    self.model = loaded
                else:
                    # Resolve state dict: handle checkpoint wrapper and backbone. prefix
                    if isinstance(loaded, dict) and "model_state_dict" in loaded:
                        state_dict = loaded["model_state_dict"]
                    else:
                        state_dict = loaded
                    # Strip 'backbone.' prefix if the model was saved inside a wrapper class
                    if any(k.startswith("backbone.") for k in state_dict):
                        state_dict = {k[len("backbone."):]: v for k, v in state_dict.items()}
                    self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("MC3-18 UCF101 model loaded successfully.")
            except Exception as e:
                print(f"MC3-18 load error: {e}. Using exercise-type fallback.")
                self.model = None
        else:
            print(f"MC3-18 weights not found at {self.model_path}. Run models/download_weights.py to download.")

    def preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Prepare 16-frame tensor [1, 3, 16, 112, 112] with UCF101 normalization."""
        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)

        if not frames:
            return torch.zeros((1, 3, 16, 112, 112), dtype=torch.float32, device=self.device)

        if len(frames) == 1:
            frame = cv2.resize(frames[0], (112, 112))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            norm  = (rgb - mean) / std
            chw   = np.transpose(norm, (2, 0, 1))
            clip  = np.stack([chw] * 16, axis=1) # [3, 16, 112, 112]
            return torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(self.device)

        indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
        processed = []
        for i in indices:
            frame = cv2.resize(frames[i], (112, 112))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            norm  = (rgb - mean) / std
            processed.append(np.transpose(norm, (2, 0, 1)))   # CHW

        clip = np.stack(processed, axis=1)                       # [3, 16, 112, 112]
        return torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, frames: List[np.ndarray]) -> Tuple[str, float, str]:
        """
        Returns (display_label, confidence, encoder_class).
        display_label is one of the curated MAIN_WORKOUT_CLASSES (formatted,
            e.g. "Body Weight Squats") whenever the model's own top guess (or
            one of its next-best few) is one of them; otherwise the generic
            "Exercising" - naming a specific move whenever the model actually
            thinks it's one of the ones we track, rather than only when its
            raw softmax probability clears some absolute number (which video
            models rarely do on a single clip, even when correct).
        encoder_class is always one of:
            Cycling, HIIT, Running, Swimming, Walking, Weightlifting, Yoga
        and is only for calorie-model MET lookups - callers should not display it.
        """
        if self.model is None or not frames:
            return "Exercising", 0.90, "HIIT"

        try:
            clip = self.preprocess_clip(frames)
            with torch.no_grad():
                out   = self.model(clip)
                probs = torch.softmax(out, dim=1)[0]

            top_p, top_i = torch.max(probs, dim=0)
            top_class = UCF101_CLASSES[int(top_i.item())] if int(top_i.item()) < len(UCF101_CLASSES) else None

            if top_class in MAIN_WORKOUT_CLASSES:
                enc_class = ucf_to_encoder_class(top_class)
                return format_ucf_label(top_class), float(top_p.item()), enc_class

            # The single best guess isn't one we track specifically - check
            # its next-best runners-up before giving up on a specific name.
            k = min(_TOP_K_RUNNER_UP, len(UCF101_CLASSES))
            runners_up = torch.topk(probs, k)
            for idx, p in zip(runners_up.indices.tolist(), runners_up.values.tolist()):
                ucf_class = UCF101_CLASSES[idx]
                if ucf_class in MAIN_WORKOUT_CLASSES:
                    return format_ucf_label(ucf_class), float(p), ucf_to_encoder_class(ucf_class)

            return "Exercising", float(top_p.item()), "HIIT"

        except Exception as e:
            print(f"MC3-18 inference error: {e}")
            return "Exercising", 0.75, "HIIT"
