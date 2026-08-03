"""
Download African wildlife model from Roboflow Universe at app startup.

Two strategies are tried in order:
1. Roboflow Python SDK  (roboflow>=1.1)
2. Direct HTTPS download via the Roboflow export API (no SDK needed)

Environment variables (set in Railway):
    ROBOFLOW_API_KEY     — required
    ROBOFLOW_WORKSPACE   — default: african-wildlife-mwx4d
    ROBOFLOW_PROJECT     — default: african-wildlife-8csiv
    ROBOFLOW_VERSION     — default: 1

The downloaded model is saved to /app/models/african-wildlife-yolov8.pt
and picked up automatically by WildlifeDetector._load_model().
"""

import logging
import os
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger("bigv.download_model")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_DEST  = MODELS_DIR / "african-wildlife-yolov8.pt"


def download() -> bool:
    # Read env vars at call time — NOT at module import time
    api_key   = os.getenv("ROBOFLOW_API_KEY", "").strip()
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "tian-jian-4ywmu")
    project   = os.getenv("ROBOFLOW_PROJECT",   "wildlife-detection-qgiwz")
    version   = os.getenv("ROBOFLOW_VERSION",   "24")

    if not api_key:
        logger.warning("ROBOFLOW_API_KEY not set — skipping Roboflow download.")
        return False

    if MODEL_DEST.exists():
        logger.info(f"Model already present at {MODEL_DEST} — skipping download.")
        return True

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Strategy 1: Roboflow Python SDK ───────────────────────────────────────
    try:
        from roboflow import Roboflow
        logger.info(f"[SDK] Downloading {workspace}/{project} v{version} ...")
        rf      = Roboflow(api_key=api_key)
        dataset = rf.workspace(workspace).project(project).version(int(version)).download(
            "yolov8", location=str(MODELS_DIR / "roboflow_tmp")
        )
        best_pt = _find_best_pt(Path(dataset.location))
        if best_pt:
            shutil.move(str(best_pt), str(MODEL_DEST))
            logger.info(f"[SDK] Model saved → {MODEL_DEST}")
            return True
        logger.warning(f"[SDK] best.pt not found under {dataset.location} — trying direct download")
    except ImportError:
        logger.info("roboflow SDK not installed — trying direct download")
    except Exception as e:
        logger.warning(f"[SDK] Failed: {e} — trying direct download")

    # ── Strategy 2: Direct HTTPS export API ───────────────────────────────────
    # Roboflow export endpoint: GET /[workspace]/[project]/[version]/yolov8/model
    url = (
        f"https://api.roboflow.com/{workspace}/{project}/{version}"
        f"/yolov8pytorch?api_key={api_key}"
    )
    logger.info(f"[HTTP] Fetching model link from Roboflow API ...")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            import json
            data = json.loads(resp.read())

        # Response contains {"model": {"link": "<signed S3 URL>"}}
        model_link = (
            data.get("model", {}).get("link")
            or data.get("export", {}).get("link")
        )
        if not model_link:
            logger.error(f"[HTTP] Unexpected API response: {list(data.keys())}")
            return False

        logger.info(f"[HTTP] Downloading weights ...")
        tmp_path = MODELS_DIR / "download_tmp.pt"
        urllib.request.urlretrieve(model_link, str(tmp_path))
        shutil.move(str(tmp_path), str(MODEL_DEST))
        logger.info(f"[HTTP] Model saved → {MODEL_DEST}")
        return True

    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        logger.error(f"[HTTP] HTTPError {e.code}: {body}")
    except Exception as e:
        logger.error(f"[HTTP] Failed: {e}")

    return False


def _find_best_pt(location: Path) -> Path | None:
    """Search common locations Roboflow SDK puts best.pt."""
    candidates = [
        location / "weights" / "best.pt",
        location / "best.pt",
        *location.rglob("best.pt"),
    ]
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(0 if download() else 1)
