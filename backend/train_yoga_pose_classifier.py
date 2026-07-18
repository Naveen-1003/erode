"""Train the yoga pose classifier (app/ai/yoga_pose_classifier.py).

Replaces the hand-tuned geometric thresholds (never validated against real
photos) with a real classifier trained on ~1000 labeled yoga pose photos from
AtomicLion1/yoga-poses-dataset (Hugging Face) - the well-known 5-class
"downdog / goddess / plank / tree / warrior2" set. Each photo is run through
this app's own MediaPipe pose pipeline (app/ai/pose_model.py) so train and
serve use the exact same landmark representation, then reduced to the
scale-invariant joint-angle feature vector in
yoga_pose_classifier.extract_pose_features() (same 14 features used at
inference time - keep both in sync if you change one).

Usage:
    python train_yoga_pose_classifier.py

Requires internet access on first run (downloads the dataset + caches it via
huggingface_hub). Writes models/yoga_pose_classifier.pkl; as soon as that file
exists, yoga_pose_classifier.classify_yoga_pose() picks it up automatically -
no code change needed.
"""

import io
import os
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app.ai.pose_model import PoseEstimator  # noqa: E402
from app.ai.yoga_pose_classifier import extract_pose_features, FEATURE_NAMES  # noqa: E402

DATASET_REPO = "AtomicLion1/yoga-poses-dataset"
DATASET_FILE = "data/train-00000-of-00001-64734869bf0f8f76.parquet"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yoga_pose_classifier.pkl")


def load_dataset() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")
    df = pd.read_parquet(path)
    df["label"] = df["file_name"].str.split("/").str[0]
    return df


def main():
    print(f"Downloading/loading {DATASET_REPO} ...")
    df = load_dataset()
    print(f"{len(df)} labeled images across classes: {sorted(df['label'].unique())}")

    pose_estimator = PoseEstimator()
    if pose_estimator.landmarker is None:
        print("MediaPipe pose landmarker not available - cannot extract training features. Aborting.")
        return

    X, y = [], []
    no_pose = 0
    for i, row in df.iterrows():
        img_field = row["image"]
        img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
        try:
            img_rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        except Exception:
            no_pose += 1
            continue
        frame_bgr = img_rgb[:, :, ::-1]  # match real inference: cv2-style BGR in, RGB conversion happens inside

        landmarks = pose_estimator.process_frame(frame_bgr)
        feats = extract_pose_features(landmarks) if landmarks else None
        if feats is None:
            no_pose += 1
            continue

        X.append(feats)
        y.append(row["label"])

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(df)} images ({len(X)} usable so far)...")

    print(f"Extracted features for {len(X)}/{len(df)} images ({no_pose} had no detectable pose and were skipped).")
    if len(X) < 50:
        print("Too few usable samples to train a meaningful classifier. Aborting.")
        return

    X = np.stack(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nHold-out test set performance:")
    print(classification_report(y_test, y_pred))

    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:28} {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "classes": list(clf.classes_)}, f)
    print(f"\nSaved trained yoga pose classifier to {MODEL_PATH}")


if __name__ == "__main__":
    main()
