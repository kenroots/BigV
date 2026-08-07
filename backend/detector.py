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
# COCO has no lion class — both "cat" and "dog" visually match lions/big cats.
# "horse" catches antelopes; "cow" catches buffalo; "sheep" catches gazelle.
COCO_AFRICAN_REMAP = {
    "horse":  "antelope",   # antelopes/impalas detected as horses
    "cow":    "buffalo",    # buffalo/wildebeest detected as cow
    "dog":    "lion",       # lions detected as dogs (body shape match) — most common in safari
    "cat":    "lion",       # lions/leopards detected as cat
    "sheep":  "gazelle",    # gazelles/springbok detected as sheep
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
    Cascade detector — runs two models and merges results for full African wildlife coverage:
      1. Roboflow wildlife-detection-xd6ml/1 — specialist: lion, cheetah, hyena, rhino, elephant, bear
      2. YOLOv8n (COCO) — generalist: giraffe, zebra, elephant, bear + COCO_AFRICAN_REMAP for others
    Results are merged; higher-confidence detection wins for any duplicate species.
    Falls back to mock if neither model loads.
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.05):
        self.confidence_threshold = confidence_threshold
        self.model      = None   # YOLOv8n (COCO)
        self.rf_client  = None   # Roboflow inference client
        self.rf_model_id = None
        self.model_name = "none"
        self.backend    = "mock"
        self.is_world_model = False
        self._load_models()

    def _load_models(self):
        """Load both Roboflow (specialist) and YOLOv8n (COCO generalist) models."""
        loaded = []

        # ── Model A: Roboflow wildlife-detection-xd6ml/1 ──────────────────────
        # Covers: Lion, Cheetah, Hyena, Elephant, Rhinoceros, Bear, Tiger
        api_key  = os.getenv("ROBOFLOW_API_KEY", "").strip()
        project  = os.getenv("ROBOFLOW_PROJECT", "wildlife-detection-xd6ml")
        version  = os.getenv("ROBOFLOW_VERSION", "1")
        workspace = os.getenv("ROBOFLOW_WORKSPACE", "psm-g8de2")
        if api_key:
            try:
                from inference_sdk import InferenceHTTPClient
                self.rf_client   = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=api_key,
                )
                self.rf_model_id = f"{project}/{version}"
                loaded.append(f"Roboflow({project} v{version})")
                logger.info(f"Roboflow model ready: {self.rf_model_id}")
            except Exception as e:
                logger.warning(f"Roboflow load failed: {e}")

        # ── Model B: YOLOv8n (COCO) ───────────────────────────────────────────
        # Covers natively: giraffe, zebra, elephant, bear, bird
        # Via COCO_AFRICAN_REMAP: lion(cat), buffalo(cow), hyena(dog), antelope(horse), gazelle(sheep)
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            loaded.append("YOLOv8n(COCO)")
            logger.info("YOLOv8n COCO model ready")
        except Exception as e:
            logger.warning(f"YOLOv8n load failed: {e}")

        if loaded:
            self.backend    = "cascade"
            self.model_name = "Cascade: " + " + ".join(loaded)
        else:
            self.backend    = "mock"
            self.model_name = "Mock Detector (demo)"
        logger.info(f"Active detector: {self.model_name}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run both models and merge results.
        Returns list of dicts: {label, confidence, bbox, priority}
        """
        if self.backend == "cascade":
            return self._detect_cascade(frame)
        elif self.backend == "ultralytics":
            return self._detect_ultralytics(frame)
        elif self.backend == "opencv_dnn":
            return self._detect_opencv_dnn(frame)
        else:
            return self._detect_mock(frame)

    # Species that COCO (yolov8n) reliably identifies — always preferred over Roboflow.
    _COCO_AUTHORITATIVE = {
        "giraffe",   # COCO native
        "zebra",     # COCO native
        "elephant",  # COCO native
        "bear",      # COCO native
        "lion",      # via cat → lion remap
        "buffalo",   # via cow → buffalo remap
        "antelope",  # via horse → antelope remap
        "gazelle",   # via sheep → gazelle remap
    }

    # Species Roboflow wildlife-detection-xd6ml/1 is authoritative for.
    # NOTE: "hyena" excluded — model frequently misclassifies lions as hyena.
    _RF_AUTHORITATIVE = {"cheetah", "rhinoceros", "tiger"}

    # Roboflow labels to suppress entirely — known misclassification sources.
    _RF_SUPPRESS = {"hyena"}

    def _detect_cascade(self, frame: np.ndarray) -> list[dict]:
        """
        Run Roboflow + YOLOv8n and merge with authority rules:
        - COCO is authoritative for giraffe/zebra/elephant/lion (via remap)
        - Roboflow is authoritative for cheetah/rhinoceros/tiger
        - For any conflict, the authoritative model wins regardless of confidence
        """
        results_rf = []
        if self.rf_client:
            try:
                # Filter out suppressed labels from Roboflow (known misclassification sources)
                results_rf = [d for d in self._detect_roboflow(frame)
                              if d["label"] not in self._RF_SUPPRESS]
            except Exception as e:
                logger.warning(f"Roboflow detection failed: {e}")

        results_coco = []
        if self.model:
            try:
                results_coco = self._detect_ultralytics(frame)
            except Exception as e:
                logger.warning(f"YOLOv8n detection failed: {e}")

        # Build label→detection maps for each source
        rf_by_label   = {d["label"]: d for d in results_rf}
        coco_by_label = {d["label"]: d for d in results_coco}
        all_labels    = set(rf_by_label) | set(coco_by_label)

        merged = []
        for label in all_labels:
            rf_det   = rf_by_label.get(label)
            coco_det = coco_by_label.get(label)

            if coco_det and label in self._COCO_AUTHORITATIVE:
                # COCO is authoritative — always use COCO detection for this species
                merged.append(coco_det)
            elif rf_det and label in self._RF_AUTHORITATIVE:
                # Roboflow is authoritative for this species
                merged.append(rf_det)
            elif coco_det and rf_det:
                # No authority rule — take higher confidence
                merged.append(coco_det if coco_det["confidence"] >= rf_det["confidence"] else rf_det)
            else:
                # Only one source has it
                merged.append(coco_det or rf_det)

        merged.sort(key=lambda d: -d["confidence"])
        logger.info(f"Cascade: RF={len(results_rf)} COCO={len(results_coco)} merged={len(merged)}"
                    + (f" → {[d['label'] for d in merged]}" if merged else ""))
        return merged

    def _detect_roboflow(self, frame: np.ndarray) -> list[dict]:
        """Send frame to Roboflow serverless inference API via SDK."""
        import cv2 as _cv2, tempfile, os as _os
        # Resize to max 416px — Roboflow free tier rejects large images
        h, w = frame.shape[:2]
        max_side = 416
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame = _cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 80])
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.tobytes())
            tmp_path = tmp.name
        try:
            raw = self.rf_client.infer(tmp_path, model_id=self.rf_model_id)
        finally:
            _os.unlink(tmp_path)

        # SDK returns a plain dict with "predictions" key
        result = raw if isinstance(raw, dict) else (vars(raw) if hasattr(raw, "__dict__") else {})

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
                original = label
                label = COCO_AFRICAN_REMAP.get(label, label)
                # Remapped labels are uncertain — require higher confidence
                # to avoid, e.g., a rhino being called "buffalo" via low-conf "cow"
                if original != label and conf < 0.30:
                    continue

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
# Fri Aug  7 11:15:55 EDT 2026
