"""
Wildlife Detector — uses YOLOv8 (ultralytics) for animal detection.
Falls back to a lightweight OpenCV DNN model if ultralytics is unavailable.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("wildlife_spotter.detector")

# COCO class names that are animals (indices from COCO dataset)
COCO_ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

# Label remaps for the COCO fallback model (yolov8n.pt).
# Only applied when no specialised model is loaded.
# NOTE: "sheep" intentionally NOT remapped — matches both warthog and gazelle.
COCO_AFRICAN_REMAP = {
    "horse":  "antelope",   # antelopes/impalas look like horses to COCO
    "cow":    "buffalo",    # buffalo/wildebeest misidentified as cow
    "dog":    "hyena",      # hyenas misidentified as dogs
    "cat":    "cheetah",    # cheetah/leopard misidentified as cat
}

# Label remaps for the active Roboflow model.
# wildlife-detection-qgiwz/24 has broken class labels (deer=buffalo/giraffe, etc.)
# Switch to a clean model via ROBOFLOW_WORKSPACE/PROJECT/VERSION env vars in Railway.
# These remaps are a temporary band-aid only.
ROBOFLOW_LABEL_REMAP: dict[str, str] = {}

# Extended wildlife labels (used with YOLO world / custom models)
WILDLIFE_LABELS = [
    "lion", "tiger", "leopard", "cheetah", "jaguar", "panther",
    "wolf", "fox", "coyote", "hyena",
    "elephant", "rhinoceros", "hippopotamus", "giraffe", "zebra",
    "buffalo", "wildebeest", "antelope", "gazelle", "impala", "springbok",
    "kudu", "eland", "oryx", "gemsbok", "hartebeest", "topi", "sable",
    "deer", "moose", "elk",
    "warthog", "wild dog", "african wild dog", "painted dog",
    "mongoose", "meerkat", "jackal",
    "bear", "grizzly bear", "polar bear", "black bear",
    "gorilla", "chimpanzee", "baboon", "monkey",
    "crocodile", "alligator", "snake", "lizard", "komodo dragon",
    "eagle", "hawk", "owl", "vulture", "flamingo", "pelican", "ostrich",
    "shark", "whale", "dolphin", "seal", "walrus",
    "kangaroo", "koala", "wombat",
    "cat", "dog", "horse", "cow", "sheep", "bird",
]

# Alert priority by animal type
ALERT_PRIORITY = {
    "lion": "critical", "tiger": "critical", "leopard": "critical",
    "cheetah": "critical", "jaguar": "critical", "panther": "critical",
    "bear": "critical", "grizzly bear": "critical", "polar bear": "critical",
    "wolf": "high", "crocodile": "critical", "alligator": "critical",
    "rhinoceros": "high", "hippopotamus": "high", "elephant": "high",
    "snake": "high", "komodo dragon": "high",
    "gorilla": "high", "chimpanzee": "medium",
    "shark": "critical",
    "wild dog": "high", "african wild dog": "high", "painted dog": "high",
    "hyena": "high",
    "warthog": "medium",
    "buffalo": "high", "wildebeest": "low",
    "antelope": "low", "gazelle": "low", "impala": "low", "springbok": "low",
    "kudu": "low", "eland": "low", "oryx": "low", "gemsbok": "low",
    "hartebeest": "low", "topi": "low", "sable": "low",
    "ostrich": "low", "mongoose": "low", "meerkat": "low", "jackal": "low",
    "default": "low",
}


class WildlifeDetector:
    """
    Animal detector supporting:
    1. Roboflow Inference SDK (serverless API) — best African wildlife accuracy
    2. YOLOv8 (ultralytics) local model — fallback
    3. OpenCV DNN with COCO model — last resort
    4. Mock detector — demo mode
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_name = "none"
        self.custom_model_path = os.getenv("WILDLIFE_MODEL_PATH", model_path)
        self._load_model(self.custom_model_path)

    def _load_model(self, model_path: Optional[str]):
        """Try Roboflow Inference SDK first, then local YOLOv8, then OpenCV DNN."""

        # ── Priority 1: Roboflow Inference SDK (serverless — no model file needed) ──
        api_key    = os.getenv("ROBOFLOW_API_KEY", "").strip()
        rf_project = os.getenv("ROBOFLOW_PROJECT", "wildlife-detection-xd6ml")
        rf_version = os.getenv("ROBOFLOW_VERSION", "4")
        if api_key:
            try:
                from inference_sdk import InferenceHTTPClient
                self.rf_client   = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=api_key,
                )
                self.rf_model_id = f"{rf_project}/{rf_version}"
                self.model_name  = f"Roboflow ({self.rf_model_id})"
                self.backend     = "roboflow_inference"
                logger.info(f"Loaded model: {self.model_name}")
                return
            except ImportError:
                logger.warning("inference-sdk not installed — falling back to local YOLOv8")
            except Exception as e:
                logger.warning(f"Roboflow Inference init failed: {e} — falling back")

        # ── Priority 2: local YOLOv8 (ultralytics) ────────────────────────────
        try:
            from ultralytics import YOLO
            # Priority: custom path > env var > default
            path = model_path or "yolov8n.pt"  # nano model — fast

            # Check models directory — Roboflow download lands here first
            models_dir = Path(__file__).parent.parent / "models"
            custom_paths = [
                models_dir / "african-wildlife-yolov8.pt",  # Roboflow download target
                models_dir / "wildlife-yolov8.pt",
                models_dir / path,
            ]

            for custom_path in custom_paths:
                if custom_path.exists():
                    path = str(custom_path)
                    logger.info(f"Found custom wildlife model: {path}")
                    break
            
            self.model = YOLO(path)
            self.model_name = f"YOLOv8 ({Path(path).name})"
            self.backend = "ultralytics"
            logger.info(f"Loaded model: {self.model_name}")
            return
        except ImportError:
            logger.warning("ultralytics not installed, trying OpenCV DNN...")
        except Exception as e:
            logger.warning(f"YOLOv8 load failed: {e}, trying OpenCV DNN...")

        # Try OpenCV DNN with YOLOv4-tiny
        try:
            models_dir = Path(__file__).parent.parent / "models"
            cfg_path = str(models_dir / "yolov4-tiny.cfg")
            weights_path = str(models_dir / "yolov4-tiny.weights")
            names_path = str(models_dir / "coco.names")

            if Path(cfg_path).exists() and Path(weights_path).exists():
                self.model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                with open(names_path) as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                self.model_name = "YOLOv4-tiny (OpenCV DNN)"
                self.backend = "opencv_dnn"
                logger.info(f"Loaded model: {self.model_name}")
                return
        except Exception as e:
            logger.warning(f"OpenCV DNN load failed: {e}")

        # Final fallback — mock detector for demo/testing
        logger.warning("No model loaded — using mock detector (demo mode)")
        self.model_name = "Mock Detector (demo)"
        self.backend = "mock"

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a frame.
        Returns list of dicts: {label, confidence, bbox, priority}
        """
        if self.backend == "roboflow_inference":
            return self._detect_roboflow(frame)
        elif self.backend == "ultralytics":
            return self._detect_ultralytics(frame)
        elif self.backend == "opencv_dnn":
            return self._detect_opencv_dnn(frame)
        else:
            return self._detect_mock(frame)

    def _detect_roboflow(self, frame: np.ndarray) -> list[dict]:
        """Send frame to Roboflow serverless inference API."""
        import cv2 as _cv2, base64, tempfile, os as _os
        # Encode frame as JPEG and send to Roboflow
        _, buf = _cv2.imencode(".jpg", frame)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.tobytes())
            tmp_path = tmp.name
        try:
            result = self.rf_client.infer(tmp_path, model_id=self.rf_model_id)
        finally:
            _os.unlink(tmp_path)

        detections = []
        for pred in result.get("predictions", []):
            conf  = float(pred.get("confidence", 0))
            if conf < self.confidence_threshold:
                continue
            raw_label = pred.get("class", "unknown").lower()
            # Apply model-specific label corrections
            label = ROBOFLOW_LABEL_REMAP.get(raw_label, raw_label)
            if raw_label != label:
                logger.info(f"Roboflow label remap: '{raw_label}' → '{label}'")
            if not self._is_animal(label):
                logger.debug(f"Roboflow skipped non-animal label: '{label}'")
                continue
            # Roboflow returns centre x/y + width/height
            cx = pred.get("x", 0); cy = pred.get("y", 0)
            bw = pred.get("width", 0); bh = pred.get("height", 0)
            detections.append({
                "label":      label,
                "confidence": round(conf, 3),
                "bbox": {
                    "x1": int(cx - bw / 2), "y1": int(cy - bh / 2),
                    "x2": int(cx + bw / 2), "y2": int(cy + bh / 2),
                },
                "priority": ALERT_PRIORITY.get(label, ALERT_PRIORITY["default"]),
            })
        return detections

    def _detect_ultralytics(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            cls_id = int(box.cls[0])
            label = results.names[cls_id].lower()

            # Only keep animals
            if not self._is_animal(label):
                continue

            # When running on the generic COCO model (yolov8n.pt), remap
            # domestic/generic labels to their African wildlife equivalents.
            # Specialised Roboflow models already output the correct labels.
            if "yolov8n" in self.model_name or "yolov8s" in self.model_name:
                label = COCO_AFRICAN_REMAP.get(label, label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "priority": ALERT_PRIORITY.get(label, ALERT_PRIORITY["default"]),
            })
        return detections

    def _detect_opencv_dnn(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        outputs = self.model.forward(output_layers)

        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                if conf < self.confidence_threshold:
                    continue
                label = self.class_names[cls_id].lower() if cls_id < len(self.class_names) else "unknown"
                if not self._is_animal(label):
                    continue
                cx, cy, bw, bh = detection[:4]
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "priority": ALERT_PRIORITY.get(label, ALERT_PRIORITY["default"]),
                })
        return detections

    def _detect_mock(self, frame: np.ndarray) -> list[dict]:
        """Demo mode: randomly simulate detections for testing UI."""
        import random
        if random.random() < 0.3:  # 30% chance of detection
            # Use the full wildlife label list so all known animals can appear
            label = random.choice(WILDLIFE_LABELS)
            h, w = frame.shape[:2]
            return [{
                "label": label,
                "confidence": round(random.uniform(0.55, 0.95), 3),
                "bbox": {
                    "x1": w // 4, "y1": h // 4,
                    "x2": 3 * w // 4, "y2": 3 * h // 4,
                },
                "priority": ALERT_PRIORITY.get(label, "low"),
            }]
        return []

    def _is_animal(self, label: str) -> bool:
        # Use whole-word matching to avoid false positives like
        # "bear" matching "wildebeest" or "cat" matching "wildcat".
        return any(w == label or label == w for w in WILDLIFE_LABELS)

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        color_map = {
            "critical": (0, 0, 255),    # Red
            "high":     (0, 100, 255),  # Orange
            "medium":   (0, 255, 255),  # Yellow
            "low":      (0, 255, 0),    # Green
        }
        annotated = frame.copy()
        for det in detections:
            bbox = det["bbox"]
            color = color_map.get(det["priority"], (255, 255, 255))
            cv2.rectangle(annotated, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), color, 2)
            label_text = f"{det['label']} {det['confidence']:.0%}"
            cv2.putText(annotated, label_text, (bbox["x1"], bbox["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated

# Made with Bob
