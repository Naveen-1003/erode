import os
import shutil
import logging
import urllib.request
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_weights")

MODELS_DIR = os.path.dirname(__file__)


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


if __name__ == "__main__":
    download_mc3_model()
    download_pose_landmarker()
