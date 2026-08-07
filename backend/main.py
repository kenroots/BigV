"""
BigV — Wildlife Spotter
Agentic Backend: FastAPI server with WebSocket live updates,
AI animal detection agent, PWA support, and alert system.
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from agent import WildlifeAgent
from detector import WildlifeDetector
from alert_manager import AlertManager
from connection_manager import ConnectionManager
from simulator import WildlifeSimulator

# ─── Logging ──────────────────────────────────────────────────────────────────
# Ensure log directory exists (use /tmp in Docker, ../logs locally)
_log_dir = Path(__file__).parent.parent / "logs"
_log_file = Path("/tmp/wildlife_spotter.log") if not _log_dir.exists() else _log_dir / "wildlife_spotter.log"
try:
    _log_dir.mkdir(exist_ok=True)
    _log_file = _log_dir / "wildlife_spotter.log"
except Exception:
    _log_file = Path("/tmp/wildlife_spotter.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_log_file)),
    ],
)
logger = logging.getLogger("wildlife_spotter")

# ─── App setup ────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """No-op lifespan — detector is initialised at module level."""
    yield

app = FastAPI(
    title="BigV — Wildlife Spotter",
    description="BigV: Agentic AI app for real-time wild animal detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

# Serve service worker at root scope (required for PWA)
@app.get("/sw.js")
async def service_worker():
    sw_path = FRONTEND_DIR / "static" / "sw.js"
    return FileResponse(str(sw_path), media_type="application/javascript",
                        headers={"Service-Worker-Allowed": "/"})

# Serve Digital Asset Links (required for TWA / Google Play)
@app.get("/.well-known/assetlinks.json")
async def asset_links():
    asset_path = FRONTEND_DIR / "static" / ".well-known" / "assetlinks.json"
    return FileResponse(str(asset_path), media_type="application/json")

# ─── Global instances ─────────────────────────────────────────────────────────
manager = ConnectionManager()
detector = WildlifeDetector()
agent = WildlifeAgent(detector)
alert_manager = AlertManager(manager)

# ─── In-memory sighting log ───────────────────────────────────────────────────
sightings: list[dict] = []

# ─── Models ───────────────────────────────────────────────────────────────────
class SightingRecord(BaseModel):
    id: str
    timestamp: str
    animals: list[dict]
    location: Optional[str] = "Unknown"
    confidence: float
    image_b64: Optional[str] = None
    alert_level: str  # "low" | "medium" | "high" | "critical"
    lat: Optional[float] = None
    lng: Optional[float] = None


class AnalyzeRequest(BaseModel):
    image_b64: str
    location: Optional[str] = "Unknown"
    camera_id: Optional[str] = "CAM-01"
    lat: Optional[float] = None
    lng: Optional[float] = None


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(), status_code=200)

@app.get("/qr", response_class=HTMLResponse)
async def qr_code_page():
    """Serve QR code page for easy app sharing."""
    qr_path = FRONTEND_DIR / "qr-code.html"
    return HTMLResponse(content=qr_path.read_text(), status_code=200)



@app.get("/health")
async def health():
    return {
        "status": "online",
        "app": "BigV",
        "detector": detector.model_name,
        "connected_clients": manager.count(),
        "total_sightings": len(sightings),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/debug/world")
async def debug_world():
    """Show current detector state and raw scores on a test frame."""
    import numpy as _np
    import asyncio as _aio

    state = {
        "detector": detector.model_name,
        "backend": detector.backend,
        "confidence_threshold": detector.confidence_threshold,
        "is_world_model": getattr(detector, "is_world_model", False),
    }

    # Run detector on a synthetic frame to surface any runtime errors
    try:
        blank = _np.zeros((320, 320, 3), dtype=_np.uint8)
        loop = _aio.get_event_loop()
        raw_boxes = await loop.run_in_executor(None, detector.detect, blank)
        state["raw_boxes_on_blank"] = raw_boxes
        state["status"] = "ok"
    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)

    return state


@app.post("/debug/raw-scan")
async def debug_raw_scan(file: UploadFile = File(...)):
    """Upload an image and return ALL YOLO boxes with no threshold filtering."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    if detector.backend != "ultralytics":
        return {"error": "ultralytics backend not active", "backend": detector.backend}

    # Show what classes the model has configured
    model_names = dict(detector.model.names) if hasattr(detector.model, "names") else {}

    results = detector.model(frame, verbose=False)[0]
    all_boxes = []
    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        all_boxes.append({"label": label, "conf": round(conf, 4)})
    # Sort by confidence descending
    all_boxes.sort(key=lambda x: -x["conf"])

    # Also try with conf=0.001 to see anything at all
    results_low = detector.model(frame, conf=0.001, verbose=False)[0]
    low_boxes = [
        {"label": results_low.names[int(b.cls[0])], "conf": round(float(b.conf[0]), 4)}
        for b in results_low.boxes
    ]
    low_boxes.sort(key=lambda x: -x["conf"])

    return {
        "image_shape": list(frame.shape),
        "model_classes": model_names,
        "total_boxes_default": len(all_boxes),
        "total_boxes_conf001": len(low_boxes),
        "current_threshold": detector.confidence_threshold,
        "is_world_model": getattr(detector, "is_world_model", False),
        "top_boxes_default": all_boxes[:10],
        "top_boxes_conf001": low_boxes[:10],
    }


@app.get("/debug/model")
async def debug_model():
    """Diagnostic endpoint — shows detector state, Roboflow env vars, and live download test."""
    from pathlib import Path
    import urllib.request, urllib.error, json as _json

    models_dir  = Path(__file__).parent.parent / "models"
    model_files = [f.name for f in models_dir.glob("*.pt")] if models_dir.exists() else []
    api_key   = os.getenv("ROBOFLOW_API_KEY", "").strip()
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "psm-g8de2")
    project   = os.getenv("ROBOFLOW_PROJECT",   "wildlife-detection-xd6ml")
    version   = os.getenv("ROBOFLOW_VERSION",   "1")

    # Probe the Roboflow inference server directly (correct endpoint)
    api_probe = "not tested"
    if api_key:
        # Use a 1x1 white JPEG as a minimal test image
        import base64
        tiny_jpg = base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAA"
            "AAAAAAcI/8QAFBABAAAAAAAAAAAAAAAAAAAAg//EABQBAQAAAAAAAAAAAAAAAAAAAAD/"
            "xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwABmX/9k="
        )
        probe_url = (
            f"https://serverless.roboflow.com/{project}/{version}"
            f"?api_key={api_key}&image={urllib.parse.quote(base64.b64encode(tiny_jpg).decode())}"
        )
        try:
            probe_url_simple = f"https://api.roboflow.com/{workspace}/{project}/{version}?api_key={api_key}"
            with urllib.request.urlopen(probe_url_simple, timeout=10) as r:
                body = _json.loads(r.read())
                api_probe = f"OK — version={body.get('version',{}).get('id','?')}"
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors='replace')[:300]
            # 404 on api.roboflow.com but 401 on serverless = model exists, key needed
            if e.code == 401:
                api_probe = "OK — model exists (401=key required, inference will work)"
            else:
                api_probe = f"HTTPError {e.code}: {body_text}"
        except Exception as e:
            api_probe = f"Error: {e}"

    return {
        "detector":           detector.model_name,
        "backend":            detector.backend,
        "models_dir":         str(models_dir),
        "model_files":        model_files,
        "roboflow_api_key":   "SET" if api_key else "NOT SET",
        "roboflow_workspace": workspace,
        "roboflow_project":   project,
        "roboflow_version":   version,
        "roboflow_api_probe": api_probe,
    }


@app.get("/debug/probe-models")
async def probe_models():
    """Probe candidate Roboflow models and return their class lists using the live API key."""
    import urllib.request as _req, urllib.parse as _parse, urllib.error as _err, json as _json, os as _os
    try:
        api_key = _os.getenv("ROBOFLOW_API_KEY", "").strip()
        if not api_key:
            return {"error": "ROBOFLOW_API_KEY not set"}

        candidates = [
            ("psm-g8de2",    "wildlife-detection-xd6ml", "1"),
            ("psm-g8de2",    "wildlife-detection-xd6ml", "2"),
            ("psm-g8de2",    "wildlife-detection-xd6ml", "3"),
            ("roboflow-100", "african-wildlife",          "4"),
            ("roboflow",     "african-wildlife",          "4"),
            ("roboflow",     "wildlife",                  "1"),
            ("roboflow",     "animals",                   "1"),
            ("roboflow",     "animals",                   "2"),
        ]

        results = []
        for ws, proj, ver in candidates:
            url = f"https://api.roboflow.com/{ws}/{proj}/{ver}?api_key={_parse.quote(api_key)}"
            try:
                with _req.urlopen(url, timeout=8) as resp:
                    data = _json.loads(resp.read())
                classes = data.get("version", {}).get("classes", [])
                results.append({"model": f"{ws}/{proj}/{ver}", "classes": classes, "count": len(classes), "status": "ok"})
            except _err.HTTPError as e:
                results.append({"model": f"{ws}/{proj}/{ver}", "status": f"HTTP {e.code}", "classes": [], "count": 0})
            except Exception as e:
                results.append({"model": f"{ws}/{proj}/{ver}", "status": f"error: {str(e)}", "classes": [], "count": 0})

        results.sort(key=lambda x: -x["count"])
        return {"candidates": results}
    except Exception as ex:
        return {"fatal_error": str(ex)}


@app.get("/debug/classes")
async def debug_classes():
    """Fetch class list directly from the Roboflow model metadata API."""
    if detector.backend != "roboflow_inference":
        return {"error": "Roboflow inference not active", "backend": detector.backend}
    import urllib.request, urllib.parse, json as _json, os as _os
    api_key   = _os.getenv("ROBOFLOW_API_KEY", "").strip()
    workspace = _os.getenv("ROBOFLOW_WORKSPACE", "psm-g8de2")
    project   = _os.getenv("ROBOFLOW_PROJECT",   "wildlife-detection-xd6ml")
    version   = _os.getenv("ROBOFLOW_VERSION",   "1")
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}?api_key={urllib.parse.quote(api_key)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read())
        classes = data.get("version", {}).get("classes", [])
        return {
            "model_id": detector.rf_model_id,
            "classes":  classes,
            "count":    len(classes),
        }
    except Exception as e:
        return {"error": str(e), "url": url.replace(api_key, "***")}


@app.post("/debug/scan")
async def debug_scan(file: UploadFile = File(...)):
    """Upload an image and get back raw Roboflow predictions with original class names."""
    if detector.backend != "roboflow_inference":
        return {"error": "Roboflow inference not active", "backend": detector.backend}
    import base64 as _b64, urllib.request, urllib.parse, json as _json, os as _os

    contents = await file.read()
    # POST image as base64 directly to the Roboflow REST API (serverless endpoint)
    b64_image = _b64.b64encode(contents).decode("utf-8")
    api_key = _os.getenv("ROBOFLOW_API_KEY", "").strip()
    url = (
        f"https://serverless.roboflow.com/{detector.rf_model_id}"
        f"?api_key={urllib.parse.quote(api_key)}"
    )
    try:
        payload = _json.dumps({"image": b64_image}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = _json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "model_id": detector.rf_model_id}

    raw_preds = result.get("predictions", [])

    from detector import ROBOFLOW_LABEL_REMAP, WILDLIFE_LABELS
    pipeline = []
    for p in raw_preds:
        raw_label = str(p.get("class", "unknown")).lower()
        remapped   = ROBOFLOW_LABEL_REMAP.get(raw_label, raw_label)
        in_list    = remapped in WILDLIFE_LABELS
        pipeline.append({
            "raw_class":     p.get("class"),
            "confidence":    round(float(p.get("confidence", 0)), 4),
            "remapped_to":   remapped,
            "passes_filter": in_list,
        })
    pipeline.sort(key=lambda x: -x["confidence"])

    return {
        "model_id":       detector.rf_model_id,
        "raw_predictions": raw_preds,
        "pipeline":        pipeline,
        "classes_seen":    sorted({p.get("raw_class") for p in pipeline}),
        "passed_filter":   [p for p in pipeline if p["passes_filter"]],
        "blocked_filter":  [p for p in pipeline if not p["passes_filter"]],
    }


@app.get("/sightings")
async def get_sightings(limit: int = 50):
    return {"sightings": sightings[-limit:], "total": len(sightings)}


@app.get("/sightings/{sighting_id}")
async def get_sighting(sighting_id: str):
    for s in sightings:
        if s["id"] == sighting_id:
            return s
    raise HTTPException(status_code=404, detail="Sighting not found")


@app.get("/community-sightings")
async def get_community_sightings():
    """Return all sightings from the last hour with GPS coords — Waze-style feed."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    recent = []
    for s in sightings:
        try:
            ts = datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))
            if ts >= cutoff:
                # Strip image data to keep payload small
                entry = {k: v for k, v in s.items() if k != "image_b64"}
                recent.append(entry)
        except Exception:
            pass
    return {
        "sightings": recent,
        "total": len(recent),
        "window_minutes": 60,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/analyze")
async def analyze_image(request: AnalyzeRequest):
    """Analyze a base64-encoded image for wildlife."""
    try:
        # Decode image
        img_data = base64.b64decode(request.image_b64.split(",")[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Run agent
        result = await agent.analyze(frame, request.location, request.camera_id)

        if result["animals"]:
            sighting = _build_sighting(result, request.image_b64, request.location,
                                       lat=request.lat, lng=request.lng)
            sightings.append(sighting)

            # Broadcast to all WebSocket clients (community update)
            await manager.broadcast(json.dumps({
                "type": "community_sighting",
                "data": {k: v for k, v in sighting.items() if k != "image_b64"},
            }))
            await manager.broadcast(json.dumps({
                "type": "sighting",
                "data": sighting,
            }))

            # Trigger alerts if needed
            await alert_manager.process(sighting)

        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    location: str = "Unknown",
    camera_id: str = "CAM-01",
    lat: Optional[float] = None,
    lng: Optional[float] = None,
):
    """Upload an image file for wildlife analysis."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    result = await agent.analyze(frame, location, camera_id)

    if result["animals"]:
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(contents).decode()
        sighting = _build_sighting(result, img_b64, location, lat=lat, lng=lng)
        sightings.append(sighting)
        await manager.broadcast(json.dumps({
            "type": "community_sighting",
            "data": {k: v for k, v in sighting.items() if k != "image_b64"},
        }))
        await manager.broadcast(json.dumps({"type": "sighting", "data": sighting}))
        await alert_manager.process(sighting)

    return result


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for live camera stream analysis."""
    await manager.connect(websocket, client_id)
    logger.info(f"Client connected: {client_id} | Total: {manager.count()}")

    # Send welcome + current stats
    await websocket.send_text(json.dumps({
        "type": "connected",
        "client_id": client_id,
        "message": "Wildlife Spotter live feed active",
        "total_sightings": len(sightings),
        "recent": sightings[-5:] if sightings else [],
    }))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                # Live frame from browser camera
                frame_b64 = msg.get("image", "")
                location = msg.get("location", "Live Camera")
                camera_id = msg.get("camera_id", client_id)
                lat = msg.get("lat")
                lng = msg.get("lng")

                try:
                    img_data = base64.b64decode(frame_b64.split(",")[-1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        result = await agent.analyze(frame, location, camera_id)

                        # Always send result back to this client
                        await websocket.send_text(json.dumps({
                            "type": "analysis",
                            "data": result,
                        }))

                        # If animals detected, broadcast sighting to all users (Waze-style)
                        if result["animals"]:
                            sighting = _build_sighting(result, frame_b64, location,
                                                       lat=lat, lng=lng,
                                                       reporter_id=client_id)
                            sightings.append(sighting)
                            # Community broadcast (no image, lightweight)
                            await manager.broadcast(json.dumps({
                                "type": "community_sighting",
                                "data": {k: v for k, v in sighting.items() if k != "image_b64"},
                            }))
                            # Full sighting (with image) to all
                            await manager.broadcast(json.dumps({
                                "type": "sighting",
                                "data": sighting,
                            }))
                            await alert_manager.process(sighting)

                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e),
                    }))

            elif msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client disconnected: {client_id} | Remaining: {manager.count()}")
        await manager.broadcast(json.dumps({
            "type": "client_left",
            "client_id": client_id,
            "connected_clients": manager.count(),
        }))


# ─── Helper ───────────────────────────────────────────────────────────────────
def _build_sighting(result: dict, image_b64: str, location: str,
                    lat: Optional[float] = None, lng: Optional[float] = None,
                    reporter_id: Optional[str] = None) -> dict:
    max_conf = max((a.get("confidence", 0) for a in result["animals"]), default=0)
    alert_level = (
        "critical" if max_conf >= 0.90 else
        "high"     if max_conf >= 0.75 else
        "medium"   if max_conf >= 0.55 else
        "low"
    )
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "animals": result["animals"],
        "location": location,
        "camera_id": result.get("camera_id", "unknown"),
        "reporter_id": reporter_id,
        "confidence": round(max_conf, 3),
        "image_b64": image_b64,
        "alert_level": alert_level,
        "agent_summary": result.get("summary", ""),
        "recommendations": result.get("recommendations", []),
        "lat": lat,
        "lng": lng,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

# Made with Bob
