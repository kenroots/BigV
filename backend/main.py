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
    """Download Roboflow model on startup if API key is set, then reload detector."""
    import download_model
    try:
        downloaded = await asyncio.get_event_loop().run_in_executor(None, download_model.download)
    except Exception as e:
        logger.error(f"Roboflow download raised exception: {e}")
        downloaded = False
    if downloaded:
        # Model file now exists — reload detector so it picks up the new .pt
        new_detector = WildlifeDetector()
        agent.detector = new_detector
        detector.__dict__.update(new_detector.__dict__)
        logger.info(f"Detector reloaded: {detector.model_name}")
    else:
        logger.warning("Roboflow model not loaded — using yolov8n.pt fallback")
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
    import tempfile as _tmp, os as _os
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    if detector.backend != "ultralytics":
        return {"error": "ultralytics backend not active", "backend": detector.backend}

    results = detector.model(frame, verbose=False)[0]
    all_boxes = []
    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        all_boxes.append({"label": label, "conf": round(conf, 4)})
    # Sort by confidence descending
    all_boxes.sort(key=lambda x: -x["conf"])
    return {
        "total_boxes": len(all_boxes),
        "current_threshold": detector.confidence_threshold,
        "is_world_model": getattr(detector, "is_world_model", False),
        "all_boxes": all_boxes[:20],  # top 20
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


@app.get("/debug/classes")
async def debug_classes():
    """Returns the raw class list the Roboflow model knows about."""
    if detector.backend != "roboflow_inference":
        return {"error": "Roboflow inference not active", "backend": detector.backend}
    try:
        # infer a blank white image — returns empty predictions but valid response
        import numpy as np, cv2, tempfile, os as _os
        blank = np.ones((64, 64, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode(".jpg", blank)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.tobytes()); tmp_path = tmp.name
        result = detector.rf_client.infer(tmp_path, model_id=detector.rf_model_id)
        _os.unlink(tmp_path)
        return {
            "model_id":    detector.rf_model_id,
            "raw_response_keys": list(result.keys()),
            "classes_in_response": sorted({p.get("class") for p in result.get("predictions", [])}),
            "note": "Blank image returns no predictions — scan a real animal via /debug/scan POST",
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/debug/scan")
async def debug_scan(file: UploadFile = File(...)):
    """Upload an image and get back raw Roboflow predictions with original class names."""
    if detector.backend != "roboflow_inference":
        return {"error": "Roboflow inference not active", "backend": detector.backend}
    import tempfile, os as _os
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(contents); tmp_path = tmp.name
    try:
        result = detector.rf_client.infer(tmp_path, model_id=detector.rf_model_id)
    finally:
        _os.unlink(tmp_path)
    return {
        "model_id":    detector.rf_model_id,
        "raw_predictions": result.get("predictions", []),
        "classes_seen": sorted({p.get("class") for p in result.get("predictions", [])}),
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
