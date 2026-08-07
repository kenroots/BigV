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

# Label remaps for wildlife-detection-xd6ml/1.
# The model uses capitalised class names and one typo — normalise to our canonical labels.
# Classes: Bear, Cheetah, Elephant, Hyena, Lion, Rhinosauras, Tiger, Wild_Boar
ROBOFLOW_LABEL_REMAP: dict[str, str] = {
    "bear":         "bear",
    "cheetah":      "cheetah",
    "elephant":     "elephant",
    "hyena":        "hyena",
    "lion":         "lion",
    "rhinosauras":  "rhinoceros",   # typo in model class name
    "tiger":        "tiger",
    "wild_boar":    "warthog",      # closest safari equivalent
}

# Extended wildlife labels — all labels YOLO-World may return
WILDLIFE_LABELS = [
    # Big cats
    "lion", "african lion", "tiger", "leopard", "cheetah", "jaguar", "panther",
    # Canids & hyenas
    "wolf", "fox", "coyote", "hyena", "spotted hyena", "african wild dog",
    "wild dog", "painted dog", "jackal",
    # Large herbivores
    "elephant", "african elephant", "rhinoceros", "black rhinoceros",
    "hippopotamus", "giraffe", "reticulated giraffe", "zebra", "plains zebra",
    # Bovids & antelope
    "buffalo", "african buffalo", "wildebeest", "antelope", "gazelle",
    "thomson gazelle", "thomson's gazelle", "thomsons gazelle",
    "grant gazelle", "grant's gazelle", "impala", "springbok",
    "kudu", "eland", "common eland", "oryx", "gemsbok", "hartebeest",
    "topi", "sable",
    # Other mammals
    "warthog", "mongoose", "meerkat",
    "gorilla", "chimpanzee", "baboon", "monkey", "colobus monkey",
    "bear", "grizzly bear", "polar bear", "black bear",
    "crocodile", "alligator", "snake", "lizard", "komodo dragon",
    # Birds
    "eagle", "hawk", "owl", "vulture", "flamingo", "pelican",
    "ostrich", "somali ostrich", "secretary bird",
    # Marine
    "shark", "whale", "dolphin", "seal", "walrus",
    # COCO fallback labels (kept so _is_animal() passes them through)
    "cat", "dog", "horse", "cow", "sheep", "bird",
    "deer", "kangaroo", "koala", "wombat",
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
    "hyena": "high", "spotted hyena": "high",
    "warthog": "medium",
    "buffalo": "high", "african buffalo": "high",
    "wildebeest": "low", "topi": "low",
    "antelope": "low", "gazelle": "low",
    "thomson gazelle": "low", "thomson's gazelle": "low", "thomsons gazelle": "low",
    "grant gazelle": "low", "grant's gazelle": "low",
    "impala": "low", "springbok": "low",
    "kudu": "low", "eland": "low", "common eland": "low",
    "oryx": "low", "gemsbok": "low", "hartebeest": "low", "sable": "low",
    "ostrich": "low", "somali ostrich": "low", "secretary bird": "low",
    "mongoose": "low", "meerkat": "low", "jackal": "low",
    "colobus monkey": "low", "baboon": "medium",
    "plains zebra": "low", "reticulated giraffe": "low",
    "african lion": "critical", "african elephant": "high",
    "black rhinoceros": "high",
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

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.05):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_name = "none"
        self.custom_model_path = os.getenv("WILDLIFE_MODEL_PATH", model_path)
        self._load_model(self.custom_model_path)

    def _load_model(self, model_path: Optional[str]):
        """Load best available detector, in priority order."""
        self.is_world_model = False

        # ── Priority 1: Roboflow Inference SDK (serverless API) ───────────────
        # Best accuracy for African wildlife — no local model file needed.
        # Activated when ROBOFLOW_API_KEY env var is set in Railway.
        api_key   = os.getenv("ROBOFLOW_API_KEY", "").strip()
        workspace = os.getenv("ROBOFLOW_WORKSPACE", "psm-g8de2")
        project   = os.getenv("ROBOFLOW_PROJECT",   "wildlife-detection-xd6ml")
        version   = os.getenv("ROBOFLOW_VERSION",   "1")
        if api_key:
            try:
                from inference_sdk import InferenceHTTPClient
                self.rf_client   = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=api_key,
                )
                self.rf_model_id = f"{project}/{version}"
                self.model_name  = f"Roboflow ({project} v{version})"
                self.backend     = "roboflow_inference"
                logger.info(f"Loaded model: {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Roboflow Inference SDK failed: {e} — falling back to YOLO-World")

        # ── Priority 2: YOLO-World open-vocabulary model ──────────────────────
        try:
            from ultralytics import YOLOWorld
            world_model = YOLOWorld("yolov8s-worldv2.pt")
            world_model.set_classes([
                # Big cats & predators
                "african lion", "leopard", "cheetah",
                # Elephants & large herbivores
                "african elephant", "black rhinoceros", "hippopotamus",
                "african buffalo", "plains zebra", "reticulated giraffe",
                # Antelope family — use both compound and simple names so
                # YOLO-World's fuzzy vocab matching has the best chance
                "wildebeest", "topi", "common eland", "oryx", "impala",
                "thomson's gazelle", "thomson gazelle", "thomsons gazelle",
                "grant's gazelle", "grant gazelle", "gazelle", "antelope",
                "springbok",
                # Canids & hyenas
                "african wild dog", "spotted hyena", "jackal",
                # Birds
                "somali ostrich", "secretary bird", "vulture", "flamingo",
                # Primates
                "colobus monkey", "baboon", "chimpanzee",
                # Other
                "warthog", "crocodile", "giraffe", "zebra",
                "lion", "elephant", "rhinoceros", "hyena", "ostrich",
            ])
            self.model      = world_model
            self.model_name = "YOLOv8-World (African wildlife)"
            self.backend    = "ultralytics"
            self.is_world_model = True
            logger.info(f"Loaded model: {self.model_name}")
            return
        except Exception as e:
            logger.warning(f"YOLO-World load failed: {e} — falling back to yolov8n")

        # ── Priority 3: local YOLOv8n (COCO fallback) ─────────────────────────
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
        """Send frame to Roboflow serverless inference API via direct REST call."""
        import cv2 as _cv2, base64 as _b64, urllib.request, urllib.parse, json as _json
        # Encode frame as JPEG → base64 string
        _, buf = _cv2.imencode(".jpg", frame)
        b64_image = _b64.b64encode(buf.tobytes()).decode("utf-8")

        api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
        url = (
            f"https://serverless.roboflow.com/{self.rf_model_id}"
            f"?api_key={urllib.parse.quote(api_key)}"
        )
        payload = _json.dumps({"image": b64_image}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = _json.loads(resp.read())

        logger.info(f"Roboflow REST response keys: {list(result.keys())}")

        raw_preds = result.get("predictions", [])
        logger.info(f"Roboflow raw predictions ({len(raw_preds)}): "
                    f"{[getattr(p, 'class', p.get('class','?')) if not isinstance(p, dict) else p.get('class','?') for p in raw_preds]}")

        detections = []
        for pred in raw_preds:
            # Normalise each prediction object
            if hasattr(pred, "dict") and callable(pred.dict):
                pred = pred.dict()
            elif hasattr(pred, "__dict__"):
                pred = vars(pred)

            conf = float(pred.get("confidence", 0))
            if conf < self.confidence_threshold:
                continue
            raw_label = str(pred.get("class", "unknown")).lower()
            # Apply model-specific label corrections
            label = ROBOFLOW_LABEL_REMAP.get(raw_label, raw_label)
            if raw_label != label:
                logger.info(f"Roboflow label remap: '{raw_label}' → '{label}'")
            if not self._is_animal(label):
                logger.info(f"Roboflow skipped non-animal/unknown label: '{label}'")
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
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]
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
            # YOLO-World already outputs correct species names — never remap it.
            if not getattr(self, "is_world_model", False):
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
