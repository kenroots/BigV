# BigV Safari — Full Deployment Flow & Tools Used

## Overview

BigV Safari is an AI-powered wildlife detection app for safari guides and tourists in Africa. It uses a **PWA (Progressive Web App)** backend wrapped in a **TWA (Trusted Web Activity)** Android shell to publish on the Google Play Store.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Android App (TWA)                     │
│         com.bigvsafari.app  (app-release-bundle.aab)    │
│         Built with: Bubblewrap CLI                       │
└────────────────────┬────────────────────────────────────┘
                     │ loads
                     ▼
┌─────────────────────────────────────────────────────────┐
│              PWA Frontend (HTML/CSS/JS)                  │
│         Served by FastAPI backend on Railway             │
│         https://bigv-production.up.railway.app           │
└────────────────────┬────────────────────────────────────┘
                     │ WebSocket + REST API
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (Python)                    │
│  - AI Wildlife Detector (YOLOv8 / mock fallback)        │
│  - Multi-agent system (detector, alert, connection mgr) │
│  - WebSocket real-time streaming                        │
│  - Serves static frontend files                         │
│  - /.well-known/assetlinks.json (Digital Asset Links)   │
└─────────────────────────────────────────────────────────┘
```

---

## Tools & Technologies Used

### Backend
| Tool | Purpose |
|------|---------|
| **Python 3.11** | Backend language |
| **FastAPI** | Web framework — REST API + WebSocket server |
| **Uvicorn** | ASGI server to run FastAPI |
| **YOLOv8 (Ultralytics)** | AI wildlife detection model (excluded from cloud build due to size) |
| **Mock Detector** | Fallback when YOLOv8 not available — simulates detections |
| **OpenCV** | Image processing for video frames |
| **WebSocket** | Real-time bidirectional communication with frontend |

### Frontend
| Tool | Purpose |
|------|---------|
| **HTML5 / CSS3 / Vanilla JS** | PWA frontend — no framework |
| **Web Manifest (`manifest.json`)** | Makes the site installable as a PWA |
| **Service Worker (`sw.js`)** | Offline caching for PWA |
| **WebSocket client** | Connects to backend for live detections |

### Cloud Deployment (Backend)
| Tool | Purpose |
|------|---------|
| **Railway** | PaaS cloud hosting for the FastAPI backend |
| **Docker** | Containerizes the backend (`Dockerfile`) |
| **GitHub** | Source code repository (`github.com/KenCher/BigV`) |
| **`railway.toml`** | Railway configuration (health check path, build settings) |

### Android App Build
| Tool | Purpose |
|------|---------|
| **Bubblewrap CLI** | Google's tool to convert a PWA into a TWA Android app |
| **Android SDK** | Required by Bubblewrap to build the Android project |
| **OpenJDK 17** | Java runtime required by Gradle (Android build system) |
| **Gradle** | Android build system (invoked by Bubblewrap) |
| **Homebrew** | Package manager used to install Node.js, OpenJDK, Android SDK on macOS |
| **Keystore (`android.keystore`)** | Signs the Android app — required for Play Store upload |

### Google Play Store
| Tool | Purpose |
|------|---------|
| **Google Play Console** | Web UI to manage and publish Android apps |
| **`.aab` (Android App Bundle)** | The upload format for Play Store (preferred over `.apk`) |
| **Digital Asset Links** | JSON file at `/.well-known/assetlinks.json` that links the Android app to the web domain — removes the URL bar in TWA |

---

## Step-by-Step Deployment Flow

### Phase 1 — Backend Development & Deployment

1. **Built FastAPI backend** (`backend/main.py`) with:
   - `/` — serves the PWA frontend HTML
   - `/static` — serves CSS, JS, images, manifest
   - `/ws` — WebSocket endpoint for real-time wildlife detections
   - `/health` — health check endpoint for Railway
   - `/.well-known/assetlinks.json` — Digital Asset Links for TWA verification

2. **Created `Dockerfile`**:
   ```dockerfile
   FROM python:3.11-slim
   RUN apt-get install -y libgl1  # OpenCV dependency
   WORKDIR /app/backend
   CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
   ```
   Key fixes:
   - Used `libgl1` (not `libgl1-mesa-glx` — removed in Debian Trixie)
   - Used `${PORT:-8000}` shell variable (Railway injects `$PORT`)
   - Set `WORKDIR /app/backend` so uvicorn finds `main.py`

3. **Created `railway.toml`**:
   ```toml
   [build]
   builder = "dockerfile"
   [deploy]
   healthcheckPath = "/health"
   ```

4. **Pushed to GitHub** → Railway auto-deployed from the repo

5. **Backend live at**: `https://bigv-production.up.railway.app`

---

### Phase 2 — Android TWA Build

1. **Installed prerequisites** via Homebrew:
   ```bash
   brew install node openjdk@17
   brew install --cask android-commandlinetools
   npm install -g @bubblewrap/cli
   ```

2. **Configured Bubblewrap** (`~/.bubblewrap/config.json`):
   ```json
   {
     "jdkPath": "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk",
     "androidSdkPath": "/opt/homebrew/share/android-commandlinetools"
   }
   ```
   Key fix: Bubblewrap config uses the JDK root (without `/Contents/Home`) — it appends that itself.

3. **Created Android SDK symlink** (Bubblewrap expects `tools/bin/sdkmanager`):
   ```bash
   mkdir -p /opt/homebrew/share/android-commandlinetools/tools/bin
   ln -s .../cmdline-tools/latest/bin/sdkmanager tools/bin/sdkmanager
   ```

4. **Ran `bubblewrap init`** in `/tmp/bigv-android/`:
   - Domain: `bigv-production.up.railway.app`
   - Package ID: `com.bigvsafari.app`
   - App name: `BigV Safari`
   - Generated `twa-manifest.json`

5. **Created keystore** with alias `android`:
   ```bash
   keytool -genkey -v -keystore android.keystore \
     -alias android -keyalg RSA -keysize 2048 \
     -validity 10000 -storepass bigvsafari123
   ```

6. **Ran `bubblewrap build`** → generated:
   - `app-release-bundle.aab` (1.2MB) — for Play Store upload
   - `app-release-signed.apk` (1.1MB) — for direct device install

7. **Extracted SHA-256 fingerprint** from keystore:
   ```bash
   keytool -list -v -keystore android.keystore -alias android \
     -storepass bigvsafari123
   ```
   Fingerprint: `00:E4:F2:B1:47:38:38:96:4C:3D:E7:96:3F:0C:24:83:68:4A:72:48:53:19:4D:9D:7A:CA:12:5D:83:BF:25:9B`

---

### Phase 3 — Digital Asset Links

1. **Created `frontend/static/.well-known/assetlinks.json`**:
   ```json
   [{
     "relation": ["delegate_permission/common.handle_all_urls"],
     "target": {
       "namespace": "android_app",
       "package_name": "com.bigvsafari.app",
       "sha256_cert_fingerprints": [
         "00:E4:F2:B1:47:38:38:96:4C:3D:E7:96:3F:0C:24:83:68:4A:72:48:53:19:4D:9D:7A:CA:12:5D:83:BF:25:9B"
       ]
     }
   }]
   ```

2. **Added route in `backend/main.py`**:
   ```python
   @app.get("/.well-known/assetlinks.json")
   async def asset_links():
       return FileResponse("../frontend/static/.well-known/assetlinks.json")
   ```

3. **Pushed to GitHub** → Railway redeployed

4. **Verified live**:
   ```bash
   curl https://bigv-production.up.railway.app/.well-known/assetlinks.json
   # Returns the JSON ✅
   ```

---

### Phase 4 — Google Play Store Upload

1. Go to **[play.google.com/console](https://play.google.com/console)**
2. **Create app** → "BigV Safari", Free, App
3. **Release → Testing → Internal testing** → Create new release
4. **Upload** `app-release-bundle.aab` from `/tmp/bigv-android/`
5. Add release notes → Save → Review → Start rollout
6. **Add testers** → Add your Gmail → Get opt-in link → Install on Android phone

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app — API, WebSocket, static file serving |
| `backend/agent.py` | Multi-agent orchestration (detector + alert agents) |
| `backend/detector.py` | YOLOv8 wildlife detector with mock fallback |
| `backend/alert_manager.py` | Alert severity classification and management |
| `frontend/index.html` | PWA main page |
| `frontend/static/js/app.js` | Frontend logic — WebSocket, Big Five tracker, UI |
| `frontend/static/css/app.css` | Safari-themed styles |
| `frontend/static/manifest.json` | PWA manifest (makes it installable) |
| `frontend/static/sw.js` | Service worker (offline support) |
| `frontend/static/.well-known/assetlinks.json` | Digital Asset Links for TWA |
| `Dockerfile` | Docker build for Railway deployment |
| `railway.toml` | Railway deployment config |
| `/tmp/bigv-android/twa-manifest.json` | Bubblewrap TWA configuration |
| `/tmp/bigv-android/android.keystore` | App signing keystore (backed up to ~/Documents/) |
| `/tmp/bigv-android/app-release-bundle.aab` | Android App Bundle for Play Store |
| `/tmp/bigv-android/app-release-signed.apk` | Signed APK for direct device install |

---

## Important Credentials to Keep Safe

| Item | Value / Location |
|------|-----------------|
| Keystore file | `~/Documents/bigv-safari-android.keystore` |
| Keystore password | `bigvsafari123` |
| Key alias | `android` |
| Package ID | `com.bigvsafari.app` |
| Railway URL | `https://bigv-production.up.railway.app` |
| GitHub repo | `https://github.com/KenCher/BigV` |

> ⚠️ **Never lose the keystore file.** It is required to publish any future updates to the app on Google Play. If lost, you cannot update the app — you would need to publish a new app with a different package ID.
---

## Alternatives to Railway for Backend Hosting

The BigV Safari backend is a **Dockerized FastAPI app** that needs:
- Persistent HTTPS URL (required for TWA + Digital Asset Links)
- WebSocket support
- Ability to run a Docker container or Python process

Here are the best alternatives, ranked by ease of use:

---

### 🥇 Render (render.com) — Best Free Alternative
| | |
|---|---|
| **Free tier** | Yes — 750 hrs/month (sleeps after 15 min inactivity) |
| **Docker support** | ✅ Yes |
| **WebSocket support** | ✅ Yes |
| **Custom domain** | ✅ Yes (free) |
| **Difficulty** | ⭐ Easy |

**How to deploy:**
1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo (`KenCher/BigV`)
3. Select **Docker** as runtime
4. Set environment variable: `PORT=8000`
5. Deploy → get URL like `https://bigv-safari.onrender.com`

**Downside:** Free tier sleeps after 15 min — first request takes ~30 seconds to wake up.

---

### 🥈 Fly.io — Best Performance Free Tier
| | |
|---|---|
| **Free tier** | Yes — 3 shared VMs, 160GB bandwidth/month |
| **Docker support** | ✅ Yes (uses your Dockerfile directly) |
| **WebSocket support** | ✅ Yes |
| **Custom domain** | ✅ Yes |
| **Difficulty** | ⭐⭐ Medium |

**How to deploy:**
```bash
brew install flyctl
flyctl auth login
cd /Users/kencheru/automation/bigv
flyctl launch        # detects Dockerfile automatically
flyctl deploy
```

**Advantage:** Does NOT sleep on free tier. Stays always-on.

---

### 🥉 Google Cloud Run — Best for Scalability
| | |
|---|---|
| **Free tier** | Yes — 2M requests/month free |
| **Docker support** | ✅ Yes (push to Google Container Registry) |
| **WebSocket support** | ✅ Yes (with session affinity) |
| **Custom domain** | ✅ Yes |
| **Difficulty** | ⭐⭐⭐ Medium-Hard |

**How to deploy:**
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/bigv-safari
gcloud run deploy bigv-safari \
  --image gcr.io/YOUR_PROJECT/bigv-safari \
  --platform managed \
  --allow-unauthenticated \
  --port 8000
```

**Advantage:** Scales to zero (no idle cost), scales up automatically under load.

---

### Heroku — Classic PaaS
| | |
|---|---|
| **Free tier** | ❌ No (removed in 2022) — starts at $5/month |
| **Docker support** | ✅ Yes |
| **WebSocket support** | ✅ Yes |
| **Custom domain** | ✅ Yes |
| **Difficulty** | ⭐ Easy |

**How to deploy:**
```bash
heroku create bigv-safari
heroku container:push web
heroku container:release web
```

---

### DigitalOcean App Platform
| | |
|---|---|
| **Free tier** | ❌ No — starts at $5/month |
| **Docker support** | ✅ Yes |
| **WebSocket support** | ✅ Yes |
| **Custom domain** | ✅ Yes |
| **Difficulty** | ⭐ Easy |

Connect GitHub repo → select Dockerfile → deploy. Very similar to Railway.

---

### VPS (DigitalOcean Droplet / Linode / Hetzner) — Most Control
| | |
|---|---|
| **Cost** | From $4/month (Hetzner) or $6/month (DigitalOcean) |
| **Docker support** | ✅ Yes (you install Docker yourself) |
| **WebSocket support** | ✅ Yes |
| **Custom domain** | ✅ Yes |
| **Difficulty** | ⭐⭐⭐⭐ Hard |

**How to deploy on a VPS:**
```bash
# On the VPS:
apt install docker.io nginx certbot
docker run -d -p 8000:8000 --env PORT=8000 your-image
# Configure nginx as reverse proxy + SSL with certbot
```

**Advantage:** Cheapest long-term, full control, no cold starts.

---

### Comparison Table

| Platform | Free Tier | Always-On | Docker | WebSocket | Difficulty |
|----------|-----------|-----------|--------|-----------|------------|
| **Railway** (current) | $5 credit | ✅ | ✅ | ✅ | ⭐ |
| **Render** | ✅ (sleeps) | ❌ | ✅ | ✅ | ⭐ |
| **Fly.io** | ✅ | ✅ | ✅ | ✅ | ⭐⭐ |
| **Google Cloud Run** | ✅ | ❌ (scales to 0) | ✅ | ✅ | ⭐⭐⭐ |
| **Heroku** | ❌ $5/mo | ✅ | ✅ | ✅ | ⭐ |
| **DigitalOcean App** | ❌ $5/mo | ✅ | ✅ | ✅ | ⭐ |
| **VPS (Hetzner)** | ❌ $4/mo | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ |

### Recommendation
- **Free + always-on** → **Fly.io**
- **Easiest migration from Railway** → **Render** or **DigitalOcean App Platform**
- **Production scale** → **Google Cloud Run** or **VPS**

---

### Migrating from Railway to Any Alternative

To switch platforms, you only need to:
1. Deploy the same Docker image to the new platform
2. Get the new HTTPS URL (e.g., `https://bigv-safari.fly.dev`)
3. Update `twa-manifest.json` → change `host` to the new domain
4. Rebuild the `.aab`:
   ```bash
   cd /tmp/bigv-android && bubblewrap build
   ```
5. Update `frontend/static/.well-known/assetlinks.json` if the domain changes (SHA-256 fingerprint stays the same — same keystore)
6. Push to GitHub → new platform redeploys
7. Upload new `.aab` to Play Console as a new release