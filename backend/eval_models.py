"""Accuracy-evaluation harness for the three action-recognition models that
have never had a local accuracy measurement: YOWOv2/AVA (yowo_action_model.py),
MC3-18/UCF101 (activity_model.py), and Kinetics-400 lifestyle
(kinetics_action_model.py). Previously only a type-shape smoke test existed
(tests/test_models.py) - this is the first real accuracy baseline for these
three, so future retrains/threshold changes can be measured against a real
before/after instead of guessed.

## Directory convention

Place short labeled video clips under:

    backend/eval_data/<model>/<class_name>/*.mp4   (.avi/.mov/.mkv also work)

where <model> is one of: mc3, yowo, kinetics
and <class_name> is one folder per ground-truth label, named however is
filesystem-safe - comparison against the model's own predicted label strips
all non-alphanumeric characters and lowercases both sides, so "Push Ups",
"push_ups", and "pushups" all compare equal. This also sidesteps AVA labels
that contain characters invalid in directory names (e.g. "run/jog" -> a
"run_jog" folder is fine).

Run `--list-classes <model>` to print the exact labels each model can
produce today, pulled live from the model's own code so this can never drift
out of sync with a real vocabulary change.

Aim for at least 10-15 clips per class to start, growing toward 15-25 over
time (5/class is too small for a trustworthy confusion matrix). Prefer
self-recorded phone clips under real deployment conditions (camera, lighting,
JPEG-ish compression) over dataset clips for this eval set specifically,
since the point is to measure what this app's users will actually see.

## Usage

    python eval_models.py --model mc3
    python eval_models.py --model all
    python eval_models.py --list-classes yowo

Results print to stdout (sklearn classification_report + confusion matrix)
and persist to backend/eval_results/<model>_<timestamp>.json so later phases
have a real baseline to diff against.
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))

from app.ai.activity_model import ActivityRecognizer, MAIN_WORKOUT_CLASSES, format_ucf_label
from app.ai.yowo_action_model import YowoActionDetector
from app.ai.kinetics_action_model import LifestyleActivityRecognizer, MAIN_LIFESTYLE_CLASSES
from app.ai.ava_action_met import get_allowed_actions
from app.services.video_processing import VideoProcessor

EVAL_DATA_ROOT = os.path.join(os.path.dirname(__file__), "eval_data")
EVAL_RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "eval_results")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
NONE_LABEL = "none_detected"  # sentinel for "model reported nothing"


def _normalize(label: str) -> str:
    """Lowercase + strip all non-alphanumeric chars, so folder names and
    model output labels compare equal regardless of spacing/case/punctuation."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _valid_classes(model: str) -> List[str]:
    if model == "mc3":
        return sorted(format_ucf_label(c) for c in MAIN_WORKOUT_CLASSES) + ["Exercising"]
    if model == "yowo":
        return sorted(get_allowed_actions())
    if model == "kinetics":
        return sorted(MAIN_LIFESTYLE_CLASSES)
    raise ValueError(f"Unknown model: {model}")


def _discover_clips(model: str) -> Dict[str, List[str]]:
    """{class_folder_name: [clip_path, ...]} for backend/eval_data/<model>/*/*"""
    root = os.path.join(EVAL_DATA_ROOT, model)
    clips_by_class: Dict[str, List[str]] = {}
    if not os.path.isdir(root):
        return clips_by_class
    for class_dir in sorted(os.listdir(root)):
        class_path = os.path.join(root, class_dir)
        if not os.path.isdir(class_path):
            continue
        clips = [
            p for p in glob.glob(os.path.join(class_path, "*"))
            if p.lower().endswith(VIDEO_EXTENSIONS)
        ]
        if clips:
            clips_by_class[class_dir] = clips
    return clips_by_class


def _sample_16(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Matches the 16-frame linspace sub-sample api/prediction.py's upload
    path applies before calling YOWOv2/Kinetics-400, so eval conditions match
    production conditions exactly."""
    idx = np.linspace(0, len(frames) - 1, min(16, len(frames))).astype(int)
    return [frames[i] for i in idx]


def _predict_mc3(recognizer: ActivityRecognizer, frames: List[np.ndarray]) -> str:
    label, _confidence, _bucket = recognizer.predict(frames)
    return label


def _predict_yowo(detector: YowoActionDetector, frames: List[np.ndarray]) -> str:
    result = detector.predict_primary_action(_sample_16(frames))
    return result["action"] if result else NONE_LABEL


def _predict_kinetics(recognizer: LifestyleActivityRecognizer, frames: List[np.ndarray]) -> str:
    result = recognizer.predict(_sample_16(frames))
    return result[0] if result else NONE_LABEL


_PREDICTORS = {
    "mc3": (ActivityRecognizer, _predict_mc3),
    "yowo": (YowoActionDetector, _predict_yowo),
    "kinetics": (LifestyleActivityRecognizer, _predict_kinetics),
}


def evaluate(model: str) -> Dict:
    if model not in _PREDICTORS:
        raise ValueError(f"Unknown model: {model}. Choose from {list(_PREDICTORS)}.")

    clips_by_class = _discover_clips(model)
    if not clips_by_class:
        print(
            f"No labeled clips found under backend/eval_data/{model}/<class>/*.mp4 - "
            f"nothing to evaluate.\n\nValid class names for '{model}':\n  "
            + "\n  ".join(_valid_classes(model))
        )
        return {}

    model_cls, predict_fn = _PREDICTORS[model]
    instance = model_cls()

    y_true: List[str] = []
    y_pred: List[str] = []
    per_clip: List[Dict] = []

    for class_name, clip_paths in clips_by_class.items():
        for clip_path in clip_paths:
            try:
                frames, _metadata = VideoProcessor.read_video(clip_path)
            except Exception as e:
                print(f"  [skip] {clip_path}: failed to read ({e})")
                continue
            if not frames:
                print(f"  [skip] {clip_path}: no frames decoded")
                continue

            predicted = predict_fn(instance, frames)
            true_norm = _normalize(class_name)
            pred_norm = _normalize(predicted)

            y_true.append(true_norm)
            y_pred.append(pred_norm)
            per_clip.append({
                "clip": clip_path,
                "true_class": class_name,
                "predicted": predicted,
                "correct": true_norm == pred_norm,
            })
            print(
                f"  {os.path.basename(clip_path):40s} true={class_name:25s} pred={predicted:25s} "
                f"{'OK' if true_norm == pred_norm else 'WRONG'}"
            )

    if not y_true:
        print(f"No clips could be read for '{model}'.")
        return {}

    report = classification_report(y_true, y_pred, zero_division=0)
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n=== {model} classification report ===")
    print(report)
    print(f"Confusion matrix labels: {labels}")
    print(np.array2string(matrix))

    os.makedirs(EVAL_RESULTS_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "model": model,
        "timestamp": timestamp,
        "num_clips": len(y_true),
        "num_classes": len(clips_by_class),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
        "per_clip": per_clip,
    }
    out_path = os.path.join(EVAL_RESULTS_ROOT, f"{model}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", choices=["mc3", "yowo", "kinetics", "all"], default="all",
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--list-classes", choices=["mc3", "yowo", "kinetics"],
        help="Print the valid class names for a model and exit",
    )
    args = parser.parse_args()

    if args.list_classes:
        print(f"Valid class names for '{args.list_classes}':")
        for c in _valid_classes(args.list_classes):
            print(f"  {c}")
        return

    models = ["mc3", "yowo", "kinetics"] if args.model == "all" else [args.model]
    for model in models:
        print(f"\n{'=' * 60}\nEvaluating {model}\n{'=' * 60}")
        evaluate(model)


if __name__ == "__main__":
    main()
