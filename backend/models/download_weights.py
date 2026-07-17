import os
import shutil
import subprocess
import logging
import urllib.request
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_weights")

MODELS_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.dirname(MODELS_DIR)


def download_mc3_model():
    repo_id = "dronefreak/mc3-18-ucf101"
    filename = "mc318-ufc101-split-1.pth"
    dest_path = os.path.join(MODELS_DIR, "mc3_ucf101.pth")

    if os.path.exists(dest_path):
        logger.info(f"MC3-18 weights already exist at {dest_path}")
        return

    logger.info(f"Downloading {filename} from Hugging Face repository {repo_id}...")
    try:
        tmp_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info(f"Downloaded to temporary path: {tmp_path}")
        shutil.copy(tmp_path, dest_path)
        logger.info(f"Successfully copied to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download MC3-18 model weights: {e}")
        raise


def download_pose_landmarker():
    dest_path = os.path.join(MODELS_DIR, "pose_landmarker_full.task")

    if os.path.exists(dest_path):
        logger.info(f"MediaPipe pose landmarker already exists at {dest_path}")
        return

    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    logger.info(f"Downloading MediaPipe pose landmarker from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Successfully downloaded pose_landmarker_full.task to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download MediaPipe pose landmarker: {e}")
        raise


def download_yowov2():
    """Clone the vendored YOWOv2 (MIT) repo + its pretrained AVA-Tiny weights.
    See app/ai/yowo_action_model.py for how it's used."""
    repo_dir = os.path.join(BACKEND_DIR, "third_party_yowov2")
    weight_path = os.path.join(repo_dir, "weights", "yowo_v2_tiny_ava.pth")

    if not os.path.isdir(repo_dir):
        logger.info("Cloning YOWOv2 (github.com/yjh0410/YOWOv2)...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/yjh0410/YOWOv2.git", repo_dir],
            check=True,
        )
    else:
        logger.info(f"YOWOv2 repo already present at {repo_dir}")

    if os.path.exists(weight_path):
        logger.info(f"YOWOv2 AVA-Tiny weights already exist at {weight_path}")
        return

    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    url = "https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_tiny_ava.pth"
    logger.info(f"Downloading YOWOv2 AVA-Tiny weights from {url}...")
    try:
        urllib.request.urlretrieve(url, weight_path)
        logger.info(f"Successfully downloaded yowo_v2_tiny_ava.pth to {weight_path}")
    except Exception as e:
        logger.error(f"Failed to download YOWOv2 weights: {e}")
        raise


if __name__ == "__main__":
    download_mc3_model()
    download_pose_landmarker()
    download_yowov2()
