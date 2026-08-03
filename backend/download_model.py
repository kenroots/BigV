"""
Download African wildlife model from Roboflow Universe at build/startup time.

Roboflow project used:
  - Workspace : african-wildlife (or set ROBOFLOW_WORKSPACE env var)
  - Project   : african-wildlife  (or set ROBOFLOW_PROJECT env var)
  - Version   : 1                 (or set ROBOFLOW_VERSION env var)

The downloaded model is saved to /app/models/african-wildlife-yolov8.pt
and picked up automatically by WildlifeDetector._load_model().

Usage:
    python download_model.py           # called from Dockerfile RUN step
    ROBOFLOW_API_KEY=xxx python download_model.py
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("bigv.download_model")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_DEST = MODELS_DIR / "african-wildlife-yolov8.pt"

# Roboflow project coordinates — override via env vars
RF_API_KEY   = os.getenv("ROBOFLOW_API_KEY", "")
RF_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "african-wildlife-mwx4d")
RF_PROJECT   = os.getenv("ROBOFLOW_PROJECT",   "african-wildlife-8csiv")
RF_VERSION   = int(os.getenv("ROBOFLOW_VERSION", "1"))


def download():
    if not RF_API_KEY:
        logger.warning(
            "ROBOFLOW_API_KEY not set — skipping Roboflow download. "
            "Set it in Railway environment variables or .env to enable African wildlife detection."
        )
        return False

    if MODEL_DEST.exists():
        logger.info(f"Model already present at {MODEL_DEST} — skipping download.")
        return True

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        return False

    logger.info(f"Downloading Roboflow model: {RF_WORKSPACE}/{RF_PROJECT} v{RF_VERSION} ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        rf = Roboflow(api_key=RF_API_KEY)
        project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        version = project.version(RF_VERSION)
        # Download as YOLOv8 PyTorch format
        dataset = version.download("yolov8", location=str(MODELS_DIR / "roboflow_tmp"))

        # Roboflow saves weights to <location>/weights/best.pt
        best_pt = Path(dataset.location) / "weights" / "best.pt"
        if not best_pt.exists():
            # Some versions put it directly
            best_pt = Path(dataset.location) / "best.pt"

        if best_pt.exists():
            best_pt.rename(MODEL_DEST)
            logger.info(f"Model saved to {MODEL_DEST}")
            return True
        else:
            logger.error(f"Could not find best.pt under {dataset.location}")
            return False

    except Exception as e:
        logger.error(f"Roboflow download failed: {e}")
        return False


if __name__ == "__main__":
    success = download()
    sys.exit(0 if success else 1)
