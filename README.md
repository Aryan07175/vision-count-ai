---
title: Vision Count AI
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# VisionCount AI 👁

> **Real-time, browser-based people counter with AI facial recognition.**  
> Detects humans via webcam, assigns each person a unique ID, and ensures nobody is counted twice — even if they leave and come back.

🚀 **[Try the Live Demo Here!](https://aaru07160-vision-count-ai.hf.space/)** 🚀

![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-COCO--SSD-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?logo=fastapi&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-Facenet512-4B0082)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ What It Does

| Feature | Description |
|---|---|
| 🔍 **Live Detection** | COCO-SSD (MobileNetV2) detects people in your webcam stream in real time |
| 🧠 **Face Recognition** | DeepFace + Facenet512 gives every person a permanent biometric ID |
| 👻 **Ghost Memory** | Remembers people who briefly step out of frame so they aren't double-counted on re-entry |
| ⚡ **Offline Fallback** | If the backend is slow or unreachable, people are still counted locally |
| 🔁 **Re-identification** | Same person walking back in hours later → same ID, no double count |
| 🎨 **Premium UI** | Glassmorphism dark theme, live stats, animated scan bars, activity log |

---

## 📸 Screenshots

The app shows a live webcam feed with:
- **Yellow box** → Person detected, scanning (building up stable frames)
- **Blue shimmer bar** → Sending to Face-ID backend for identification
- **Teal box** → Person counted and assigned an ID
- **Purple ring** → Ghost (person recently left frame, still remembered)

---

## 🏗️ Architecture

```
Browser (React + Vite)               Python Backend (FastAPI)
┌─────────────────────────┐          ┌──────────────────────────┐
│  Webcam → TF.js COCO-SSD│          │  /identify               │
│  Centroid tracking       │ ──POST──▶│  DeepFace Facenet512     │
│  Ghost memory registry   │ ◀─JSON── │  Face embedding store    │
│  UI counter + logs       │          │  /reset                  │
└─────────────────────────┘          └──────────────────────────┘
        ↑ Vite proxy strips /api prefix and forwards to :8000
```

---

## 🛠️ Tech Stack

### Frontend
| Tool | Purpose |
|---|---|
| React 18 + Vite 8 | UI framework and dev server |
| TensorFlow.js + COCO-SSD | In-browser person detection (MobileNetV2 backbone) |
| Vanilla CSS | Glassmorphism dark theme, animations |
| `react-webcam` | Camera access |

### Backend
| Tool | Purpose |
|---|---|
| FastAPI + Uvicorn | HTTP API server |
| DeepFace (Facenet512) | 512-dimension face embedding & matching |
| OpenCV + NumPy | Image decoding and preprocessing |
| Python 3.11+ | Runtime |

---

## ⚡ Quick Start

### Requirements
- **Node.js** v18 or newer
- **Python** 3.9 or newer
- A webcam

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/Aryan07175/vision-count-ai.git
cd vision-count-ai
```

---

### Step 2 — Set up the Python backend

```bash
# Go into the backend folder
cd backend

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```

> 💡 **First-run note:** DeepFace will automatically download the Facenet512 model weights (~90 MB) the first time it runs. This is a one-time download.

Start the backend:
```bash
# From the project root (not the backend folder)
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The backend is now running at **http://localhost:8000**

---

### Step 3 — Set up the frontend

Open a **new terminal** in the project root:

```bash
# Install Node dependencies
npm install

# Start the Vite dev server
npm run dev
```

The app is now running at **http://localhost:5173**

---

### Step 4 — Open the app

1. Go to **http://localhost:5173** in your browser
2. **Allow camera access** when the browser asks
3. Wait a few seconds for `AI Ready` to appear in the top-right
4. Walk in front of the camera — you will be detected, identified, and counted!

---

## 🔄 How It Works (Step by Step)

```
1. DETECT   ─ COCO-SSD scans each webcam frame for people (every 150 ms)
               Yellow box appears, scan bar starts filling

2. STABILISE─ Person must appear for 4+ consecutive frames before identify fires
               (Prevents wasted API calls on brief detections)

3. IDENTIFY ─ A crop of the person is sent to the FastAPI backend
               Blue shimmer bar plays while waiting for the response

4. MATCH    ─ Backend extracts a Facenet512 embedding and compares to known faces
               • New face  → Assigned a permanent ID, counter increments by 1
               • Known face → Logged as "Already Detected", counter stays the same

5. GHOST    ─ If the person leaves frame, their position is remembered for 3 minutes
               Re-entry within that window → same ID, no double count

6. FALLBACK ─ If the backend times out (3 attempts), the person is counted locally
               with a temporary L-ID. Upgrades to a real Face-ID if backend recovers.
```

---

## 🎛️ Key Settings (in `PeopleCounter.jsx`)

| Constant | Default | What it controls |
|---|---|---|
| `MIN_FRAMES` | `4` | Frames a person must be stable before identify fires |
| `MAX_DISAPPEARED` | `30` | Frames before a track moves to ghost memory (~4.5 s) |
| `GHOST_TTL_MS` | `180000` | How long ghost memory lasts (3 real minutes) |
| `MAX_NO_FACE_ATTEMPTS` | `3` | Failed face attempts before falling back to local ID |
| `IDENTIFY_TIMEOUT` | `6000` | Max ms to wait for the backend before timeout |
| `DETECT_MS` | `150` | Detection interval in milliseconds |

---

## 🐛 Bug Fixes (v2 — June 2026)

This version includes a full audit and fixes of 13 issues:

| # | Severity | Fix |
|---|---|---|
| 1 | 🔴 Critical | `localNextPersonId` reset to `0` on Reset — would clash with backend Face-IDs (now resets to `10000`) |
| 2 | 🔴 Critical | `window.countAnimTimeout` global leak replaced with `useRef` — prevents setState-on-unmounted warnings |
| 3 | 🟡 Logic | `continue` after async `.then()` refactored to a clear `if/else` guard clause |
| 4 | 🟡 Logic | Missing outer `else` branch — `appeared` counter was frozen at 1, identify never fired, count stayed 0 |
| 5 | 🟡 Logic | Tracking refs hoisted to module-level singletons — survives React 18 StrictMode double-mount |
| 6 | 🟡 Logic | Scan bar now animates with a pulsing shimmer during Face-ID identify phase (was frozen at 100%) |
| 7 | 🟡 Logic | Ghost TTL now uses `Date.now()` timestamps instead of frame counts (immune to CPU slowdowns) |
| 8 | 🟡 Perf | Explicit `tf.setBackend('webgl')` added — previously could silently fall back to CPU (10–20× slower) |
| 9 | 🟡 Logic | `MAX_NO_FACE_ATTEMPTS` raised from `1` → `3` — prevents instant overcounting on brief occlusions |
| 10 | 🟡 Backend | `asyncio.get_event_loop()` → `asyncio.get_running_loop()` (deprecated in Python 3.10+) |
| 11 | 🟡 Backend | `threading.Lock` added around `next_id` — prevents duplicate IDs under concurrent requests |
| 12 | 🟢 Perf | TF.js + COCO-SSD now dynamically imported — initial JS bundle reduced from **1.3 MB → 217 KB (−83%)** |
| 13 | 🟢 Minor | Inter font now actually loaded via Google Fonts `<link>` in `index.html` |

---

## 📁 Project Structure

```
vision-count-ai/
├── backend/
│   ├── app.py              # FastAPI server — face embedding & matching logic
│   ├── requirements.txt    # Pinned Python dependencies
│   └── venv/               # Python virtual environment (not committed)
│
├── src/
│   ├── components/
│   │   ├── PeopleCounter.jsx   # Main component — detection, tracking, UI
│   │   └── PeopleCounter.css   # All component styles
│   ├── App.jsx
│   └── main.jsx
│
├── index.html          # Root HTML — font imports, meta tags
├── vite.config.js      # Vite config — dev server + /api proxy to :8000
└── package.json
```

---

## 🔧 Troubleshooting

### Counter stays at 0
- Make sure the backend is running at **port 8000**
- Check the top-right status pill: `Face-ID Active` (green) = backend connected, `Local Mode ⚡` (yellow) = backend offline
- On first run, wait 60–90 seconds for DeepFace to download its model weights
- Click **🔄 Reset** and try again after the backend is fully ready

### Camera not showing
- Make sure you clicked **Allow** when the browser asked for camera permission
- Try clicking **🔁 Switch Camera** to toggle between front/back camera
- Check that no other app is using the camera

### DeepFace timeout / Local Mode
- DeepFace is slow on the first request because it loads the model into memory
- After the first successful identification it becomes much faster
- The app automatically falls back to local counting so no detections are missed

### Backend import error on startup
- Make sure you activated the `venv` before running uvicorn
- Run from the **project root**, not the `backend/` folder
- Command: `uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload`

---

## 📄 License

MIT — free to use, modify, and distribute.
