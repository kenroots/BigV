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


def download():
    # Read env vars inside the function so Railway runtime vars are visible
    api_key   = os.getenv("ROBOFLOW_API_KEY", "").strip()
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "african-wildlife-mwx4d")
    project   = os.getenv("ROBOFLOW_PROJECT",   "african-wildlife-8csiv")
    version   = int(os.getenv("ROBOFLOW_VERSION", "1"))

    if not api_key:
        logger.warning(
            "ROBOFLOW_API_KEY not set — skipping Roboflow download. "
            "Set it in Railway environment variables to enable African wildlife detection."
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

    logger.info(f"Downloading Roboflow model: {workspace}/{project} v{version} ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        rf = Roboflow(api_key=api_key)
        dataset = rf.workspace(workspace).project(project).version(version).download(
            "yolov8", location=str(MODELS_DIR / "roboflow_tmp")
        )

        # Roboflow saves weights to <location>/weights/best.pt
        best_pt = Path(dataset.location) / "weights" / "best.pt"
        if not best_pt.exists():
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
