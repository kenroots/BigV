# 🦁 Wildlife Spotter — Agentic AI Real-Time Detection App

A full-stack agentic application for **real-time wild animal detection** using your camera or uploaded images. Powered by **YOLOv8**, **FastAPI**, and **WebSockets** for live updates.

---

## 🚀 Quick Start

```bash
cd /Users/kencheru/automation/wildlife-spotter
bash start.sh --install   # first time (installs deps)
bash start.sh             # subsequent runs
```

Then open: **http://localhost:8000**

---

## 🏗️ Architecture

```
wildlife-spotter/
├── backend/
│   ├── main.py              # FastAPI server + WebSocket endpoints
│   ├── agent.py             # 🤖 Agentic AI layer (reasoning, memory, summaries)
│   ├── detector.py          # 🔍 YOLOv8 animal detector (falls back to mock)
│   ├── alert_manager.py     # 🚨 Alert system (WebSocket push + Slack)
│   └── connection_manager.py# 🔌 WebSocket connection pool
├── frontend/
│   ├── index.html           # Main UI
│   └── static/
│       ├── css/app.css      # Dark safari theme
│       └── js/app.js        # Live camera + WebSocket client
├── models/                  # Optional: custom YOLO weights
├── logs/                    # Server logs
├── requirements.txt
├── start.sh                 # One-command startup
└── .env.example             # Config template
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Live Camera Feed** | Browser webcam capture with real-time frame analysis |
| 🤖 **AI Agent** | YOLOv8-powered detection + reasoning layer with memory |
| 🔍 **Animal Detection** | 80+ COCO classes + extended wildlife labels |
| 📊 **Danger Scoring** | 0–100% danger score per frame based on animal type |
| 🚨 **Live Alerts** | WebSocket push alerts with priority levels (low/medium/high/critical) |
| 🖼️ **Image Upload** | Analyze static images in addition to live feed |
| 📋 **Agent Summaries** | Natural language descriptions + safety recommendations |
| 🔔 **Browser Notifications** | Desktop push notifications for high-priority sightings |
| 📜 **Sightings Log** | Scrollable history with thumbnails |
| 📡 **Multi-client** | Multiple browser tabs/users see the same live alerts |
| 💬 **Slack Alerts** | Optional Slack webhook for critical/high alerts |

---

## 🐾 Detected Animals

**Critical priority:** Lion, Tiger, Leopard, Cheetah, Bear, Crocodile, Alligator, Shark  
**High priority:** Wolf, Elephant, Rhinoceros, Hippopotamus, Snake, Komodo Dragon, Gorilla  
**Medium priority:** Chimpanzee, Baboon, Monkey  
**Low priority:** Deer, Fox, Eagle, Bird, Horse, Cow, Sheep, and all other COCO animals

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `SLACK_WEBHOOK_URL` | Slack webhook for critical alerts | (none) |
| `CONFIDENCE_THRESHOLD` | Detection confidence (0–1) | `0.45` |
| `YOLO_MODEL_PATH` | Custom YOLO model path | `yolov8n.pt` |
| `PORT` | Server port | `8000` |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Server health + stats |
| `GET` | `/sightings` | All sightings log |
| `POST` | `/analyze` | Analyze base64 image |
| `POST` | `/upload` | Upload image file |
| `WS` | `/ws/{client_id}` | Live WebSocket stream |

---

## 🎮 Usage

### Live Camera
1. Open http://localhost:8000
2. Click **▶ Start** — allow camera access
3. Set your location (e.g. "Serengeti North Gate")
4. Watch real-time detections with bounding boxes

### Upload Image
1. Click **📁 Upload Image**
2. Select any photo
3. Agent analyzes and shows results instantly

### Keyboard Shortcuts
- `Space` — Start/Stop camera
- `Esc` — Close alert modal

---

## 🤖 AI Agent Capabilities

The **WildlifeAgent** layer adds intelligence on top of raw detection:

- **Memory**: Tracks last 20 detections per camera — detects repeat sightings
- **Behavior inference**: Identifies group movement, solitary movement, foraging
- **Danger scoring**: Weighted score based on animal type × confidence
- **Natural language summaries**: Human-readable descriptions of each frame
- **Contextual recommendations**: Safety actions based on detected species
- **🌍 Location awareness**: Filters animals by region (East/Southern/West/Central Africa)
- **🏞️ Park-specific data**: Recognizes 8 major safari parks with expected wildlife
- **⚠️ Unusual sighting alerts**: Flags animals not typically found in that location
- **🏆 Big Five tracking**: Monitors progress toward seeing all Big Five animals
- **📚 Safari knowledge base**: Swahili names, habitats, behaviors, photo tips

---

## 📦 Models

### Standard YOLOv8 Models

| Model | Size | Speed | Accuracy | Auto-download |
|-------|------|-------|----------|---------------|
| `yolov8n.pt` | 6MB | ⚡ Fast | Good | ✅ Yes |
| `yolov8s.pt` | 22MB | Fast | Better | Manual |
| `yolov8m.pt` | 52MB | Medium | Best | Manual |

### 🌍 African Wildlife Models (Recommended)

For **better accuracy on African safari animals**, use specialized models:

#### Option 1: Roboflow Universe (Easiest)
1. Visit [Roboflow Universe](https://universe.roboflow.com/)
2. Search for: `"African wildlife"`, `"Big Five"`, or `"Safari animals"`
3. Download YOLOv8 format (`.pt` file)
4. Place in `models/` directory
5. Set in `.env`:
   ```bash
   WILDLIFE_MODEL_PATH=models/african-wildlife-yolov8.pt
   ```

#### Option 2: Pre-trained Wildlife Models
- **MegaDetector** (Microsoft): General wildlife detection from camera traps
- **Wildlife Insights**: Conservation-focused models
- **iNaturalist**: Species identification models

#### Option 3: Train Your Own
```bash
# Using African wildlife dataset
yolo train data=african-wildlife.yaml model=yolov8n.pt epochs=100
# Model saved to runs/detect/train/weights/best.pt
cp runs/detect/train/weights/best.pt models/african-wildlife-yolov8.pt
```

**Datasets for training:**
- Kaggle: "African Wildlife Dataset"
- LILA BC: Camera trap images
- Your own camera trap footage

### 🗺️ Location-Based Filtering

The app now includes **regional wildlife filtering** for 8 major safari parks:

| Park | Country | Common Animals |
|------|---------|----------------|
| **Serengeti** | Tanzania | Lion, Leopard, Cheetah, Elephant, Buffalo, Wildebeest, Zebra |
| **Maasai Mara** | Kenya | Lion, Leopard, Cheetah, Elephant, Buffalo, Wildebeest, Zebra |
| **Kruger** | South Africa | Lion, Leopard, Cheetah, Elephant, Buffalo, Rhinoceros |
| **Okavango** | Botswana | Lion, Leopard, Elephant, Hippopotamus, Crocodile |
| **Amboseli** | Kenya | Elephant, Lion, Cheetah, Buffalo, Zebra, Giraffe |
| **Ngorongoro** | Tanzania | Lion, Leopard, Elephant, Buffalo, Rhinoceros |
| **Chobe** | Botswana | Elephant, Buffalo, Lion, Hippopotamus, Crocodile |
| **Hwange** | Zimbabwe | Elephant, Lion, Leopard, Buffalo, Giraffe |

**How it works:**
- Enter location name (e.g., "Serengeti", "Kruger National Park")
- App filters detections to show expected vs. unexpected animals
- Alerts you to unusual sightings for that region
- Provides location-specific recommendations

### Using Custom Models

To use a larger or custom model:
```bash
# Standard YOLOv8 model
export WILDLIFE_MODEL_PATH=yolov8m.pt
bash start.sh

# African wildlife model
export WILDLIFE_MODEL_PATH=models/african-wildlife-yolov8.pt
bash start.sh
```

Or set in `.env` file (see `.env.example` for details).

---

## 🔒 Security

- No credentials are hardcoded
- All secrets via environment variables / `.env` file
- `.env` is in `.gitignore` — never committed
- Pre-commit hook installed to block accidental secret commits