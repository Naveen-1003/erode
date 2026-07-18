"""Train the walking classifier (app/ai/walking_classifier.py).

Replaces the hand-tuned anti-phase correlation threshold (never validated
against real gait data) with a real classifier trained on real HMDB51 video
clips - specifically kiyoonkim/hmdb-51-posec3d on Hugging Face, which ships
HMDB51 already reduced to per-frame COCO-17 keypoint sequences (via the
MMAction2 PoseC3D pipeline), so no video decoding or pose estimation needs to
happen here.

"walk" (264 clips) is the positive class. Negatives are a deliberately mixed
set of other HMDB51 actions, chosen to force the classifier to learn the
actual alternating-gait signature rather than "any repetitive motion" or
"any leg motion":
    - run: hardest negative - another cyclic leg gait, just faster/higher
      amplitude. If the classifier can't tell these apart, that's fine (both
      are "the person is locomoting"); we mainly care it doesn't fire on the
      classes below.
    - climb_stairs: also alternates the legs, but with a very different
      vertical/knee-lift profile than level walking.
    - stand, sit: static poses, no gait at all.
    - jump, pushup, situp: rhythmic but BOTH-legs-together (in-phase), unlike
      walking's alternating pattern.
    - wave, clap, punch: upper-body-only motion, legs mostly still.

Each clip is reduced to the SAME 7-feature vector used at inference time
(walking_classifier.extract_gait_features) via a sliding window over its
frames - keep the two in sync if you change one.

Usage:
    python train_walking_classifier.py

Requires internet access on first run (downloads the pkl via
huggingface_hub). Writes models/walking_classifier.pkl; as soon as that file
exists, walking_classifier.is_walking() picks it up automatically - no code
change needed.
"""

import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app.ai.walking_classifier import (  # noqa: E402
    extract_gait_features, _make_simple_frame, FEATURE_NAMES, WINDOW,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "walking_classifier.pkl")

# Standard 51-class alphabetical HMDB51 label order (verified empirically
# against this dataset's frame_dir names - see conversation/investigation).
HMDB51_CLASSES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive",
    "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand",
    "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push",
    "pushup", "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun",
    "sit", "situp", "smile", "smoke", "somersault", "stand", "swing_baseball", "sword", "sword_exercise",
    "talk", "throw", "turn", "walk", "wave",
]

POSITIVE_CLASS = "walk"
NEGATIVE_CLASSES = [
    "run", "climb_stairs", "stand", "sit", "jump", "pushup", "situp", "wave", "clap", "punch",
]

# COCO-17 keypoint indices (this dataset's format)
_L_SH, _R_SH = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16
_MIN_JOINT_SCORE = 0.3  # drop frames where any tracked joint's confidence is below this

STRIDE = 8  # overlapping windows every STRIDE frames within a clip


def load_hmdb51_posec3d():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="kiyoonkim/hmdb-51-posec3d", filename="hmdb51_2d.pkl", repo_type="dataset")
    with open(path, "rb") as f:
        return pickle.load(f)


def clip_to_simple_frames(ann) -> list:
    """One HMDB51 clip's primary-person COCO-17 keypoints -> a list of
    _make_simple_frame() dicts, dropping frames with weak joint confidence."""
    kp = ann["keypoint"][0]          # [T, 17, 2] - primary detected person
    score = ann["keypoint_score"][0]  # [T, 17]

    frames = []
    for t in range(kp.shape[0]):
        joint_scores = score[t, [_L_SH, _R_SH, _L_HIP, _R_HIP, _L_KNEE, _R_KNEE, _L_ANKLE, _R_ANKLE]]
        if joint_scores.min() < _MIN_JOINT_SCORE:
            continue
        frames.append(_make_simple_frame(
            tuple(kp[t, _L_SH]), tuple(kp[t, _R_SH]),
            tuple(kp[t, _L_HIP]), tuple(kp[t, _R_HIP]),
            tuple(kp[t, _L_KNEE]), tuple(kp[t, _R_KNEE]),
            tuple(kp[t, _L_ANKLE]), tuple(kp[t, _R_ANKLE]),
        ))
    return frames


def clip_to_windowed_features(ann) -> list:
    """Sliding-window feature vectors for one clip (may be zero if the clip
    is too short or too much of it has low-confidence tracking)."""
    frames = clip_to_simple_frames(ann)
    out = []
    for start in range(0, max(1, len(frames) - WINDOW + 1), STRIDE):
        window = frames[start:start + WINDOW]
        feats = extract_gait_features(window)
        if feats is not None:
            out.append(feats)
    return out


def main():
    print("Loading kiyoonkim/hmdb-51-posec3d ...")
    data = load_hmdb51_posec3d()
    anns = data["annotations"]

    target_names = {POSITIVE_CLASS} | set(NEGATIVE_CLASSES)
    target_labels = {HMDB51_CLASSES.index(n): n for n in target_names}
    print(f"Target classes: positive={POSITIVE_CLASS!r}, negatives={NEGATIVE_CLASSES}")

    X, y, clip_ids = [], [], []
    for clip_idx, ann in enumerate(anns):
        name = target_labels.get(ann["label"])
        if name is None:
            continue
        label = 1 if name == POSITIVE_CLASS else 0
        for feats in clip_to_windowed_features(ann):
            X.append(feats)
            y.append(label)
            clip_ids.append(clip_idx)  # so train/test split keeps a clip's windows together

    X = np.stack(X)
    y = np.array(y)
    clip_ids = np.array(clip_ids)
    print(f"{len(X)} windows total ({int(y.sum())} positive / {len(y) - int(y.sum())} negative) "
          f"from {len(set(clip_ids))} clips.")

    # Split by CLIP, not by window, so windows from the same clip never
    # leak across train/test (they're highly correlated with each other).
    unique_clips = np.array(sorted(set(clip_ids)))
    train_clips, test_clips = train_test_split(unique_clips, test_size=0.25, random_state=0)
    train_mask = np.isin(clip_ids, train_clips)
    test_mask = np.isin(clip_ids, test_clips)

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=0, class_weight="balanced")
    clf.fit(X[train_mask], y[train_mask])

    y_pred = clf.predict(X[test_mask])
    print("\nHold-out test set performance (split by clip, not by window):")
    print(classification_report(y[test_mask], y_pred, target_names=["not_walk", "walk"]))

    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:20} {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf}, f)
    print(f"\nSaved trained walking classifier to {MODEL_PATH}")


if __name__ == "__main__":
    main()
