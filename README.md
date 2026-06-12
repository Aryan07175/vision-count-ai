---
title: Vision Count AI
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

<div align="center">

# 👁‍🗨 VisionCount AI

**Real-time AI-powered people detection, tracking & counting — right in your browser.**

[![CI — Build & Lint](https://github.com/Aryan07175/vision-count-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Aryan07175/vision-count-ai/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-orange?logo=huggingface)](https://aaru07160-vision-count-ai.hf.space/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Node](https://img.shields.io/badge/Node-20-brightgreen?logo=node.js)](package.json)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](backend/requirements.txt)

🚀 **[Try the Live Demo →](https://aaru07160-vision-count-ai.hf.space/)**

*(Allow camera access when your browser asks — it's needed to detect people!)*

</div>

---

## 🎯 What Problem Does This Solve?

Most people-counting systems have a critical flaw: **they count the same person multiple times**. Walk in, walk out, walk back in — that's counted as 3 people instead of 1.

VisionCount AI solves this with two layers of intelligence:
- **Body tracking** (TensorFlow.js, in-browser) to follow movement frame-by-frame
- **Face identity** (DeepFace, on server) to assign each person a permanent unique ID so they're never double-counted

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Real-time detection** | COCO-SSD detects people at 150 ms/frame with NMS deduplication |
| 🧠 **Face recognition** | Facenet512 model assigns a permanent ID per face |
| 👻 **Ghost memory** | Tracks people who leave the frame; re-matches on return |
| 🔁 **No double counting** | Same person seen multiple times = counted exactly once |
| 📡 **Offline fallback** | Backend unreachable? Falls back to anonymous ID counting |
| 📱 **Mobile-ready** | Capacitor support for native iOS/Android builds |
| 🌐 **Zero install for users** | Runs fully in-browser — no app download needed |

---

## 🏗️ Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER'S BROWSER                              │
│                                                                      │
│  Webcam Feed ──► TensorFlow.js (COCO-SSD)                           │
│                       │                                              │
│                  Person detected?                                    │
│                       │                                              │
│                  NMS filter ──► Centroid Tracker                    │
│                       │              │                               │
│              New/unidentified     Matched to                        │
│              person (5+ frames)   existing track                     │
│                       │                                              │
│                  Crop face region                                    │
│                       │                                              │
└───────────────────────┼──────────────────────────────────────────────┘
                        │ HTTP POST /api/identify (base64 JPEG crop)
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                               │
│                                                                      │
│  Receive image ──► SHA256 cache lookup (4s TTL LRU cache)           │
│                       │                                              │
│                  Cache miss?                                         │
│                       │                                              │
│            DeepFace.represent()                                      │
│            (Facenet512, SSD → OpenCV fallback)                      │
│                       │                                              │
│            Cosine distance vs. known faces                           │
│                       │                                              │
│         Match < 0.34?──► Return existing ID ("recognized")          │
│         No match?    ──► Assign new ID, store embedding ("new")     │
│         No face?     ──► Return null ("no_face")                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                        │ JSON { id, status, distance }
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BROWSER (continued)                            │
│                                                                      │
│  Backend ID returned ──► Mark track as counted, log to activity     │
│  Backend offline     ──► Assign Anon-X ID, continue counting        │
│  No face detected    ──► Keep trying for 4.5 s, then assign Anon-X  │
│                                                                      │
│  Canvas overlay: corner brackets, ID badge, scan progress bar       │
│  Session timer, live "In Frame" count, ghost memory counter         │
└─────────────────────────────────────────────────────────────────────┘
```

### CI/CD Pipeline

```
Developer pushes code
        │
        ▼
┌───────────────────────┐
│  GitHub Actions CI    │
│  (on push / PR)       │
│                       │
│  ┌─────────────────┐  │
│  │  Frontend Job   │  │
│  │  npm ci         │  │
│  │  npm run lint   │  │
│  │  npm run build  │  │
│  │  Upload dist/   │  │
│  └─────────────────┘  │
│  ┌─────────────────┐  │
│  │  Backend Job    │  │
│  │  pip install    │  │
│  │  py_compile     │  │
│  │  import check   │  │
│  └─────────────────┘  │
└───────────────────────┘
        │ All green ✅
        ▼
  PR review & merge → main
        │
        ▼
┌───────────────────────┐
│  Hugging Face Spaces  │
│  Docker build         │
│  Stage 1: npm build   │
│  Stage 2: pip install │
│  Expose port 7860     │
└───────────────────────┘
```

---

## 💻 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19 + Vite | UI, webcam access, canvas overlay |
| **In-Browser AI** | TensorFlow.js + COCO-SSD | Real-time person detection |
| **Tracker** | Custom centroid + NMS | Frame-to-frame tracking, ghost memory |
| **Backend API** | FastAPI + Uvicorn | REST API, request serialization |
| **Face AI** | DeepFace (Facenet512) | Face embedding & identity matching |
| **Image Processing** | OpenCV + NumPy | CLAHE enhancement, unsharp masking |
| **Mobile** | Capacitor | Native iOS/Android wrapper |
| **Deployment** | Docker + Hugging Face Spaces | Cloud hosting, port 7860 |
| **CI/CD** | GitHub Actions | Lint, build, type-check on every push |

---

## 🚀 Local Development

### Prerequisites

- **Node.js** ≥ 20
- **Python** ≥ 3.10
- A webcam

### 1. Clone the repo

```bash
git clone https://github.com/Aryan07175/vision-count-ai.git
cd vision-count-ai
```

### 2. Start the Python Backend

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies (first run downloads AI models — ~1–2 min)
pip install -r backend/requirements.txt

# Start the server
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** The very first run will download Facenet512 model weights (~100 MB). Subsequent starts are instant.

### 3. Start the Frontend

Open a **new terminal** in the project root:

```bash
npm install
npm run dev
```

Open **http://localhost:5173** in your browser, allow camera access, and step in front of your webcam!

### 4. (Optional) Configure Mobile Backend URL

If you're building for Capacitor (native mobile), copy `.env.example` to `.env` and set your machine's LAN IP:

```bash
cp .env.example .env
# Edit .env and set:
# VITE_BACKEND_URL=http://YOUR_LOCAL_IP:8000
```

---

## 🐳 Docker (Production)

The app ships as a single Docker image — frontend static files served by the FastAPI backend.

```bash
# Build
docker build -t vision-count-ai .

# Run (Hugging Face Spaces uses port 7860)
docker run -p 7860:7860 vision-count-ai
```

Open **http://localhost:7860**.

---

## 📁 Project Structure

```
vision-count-ai/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── backend/
│   ├── app.py                  # FastAPI app — face identity API
│   └── requirements.txt        # Python dependencies
├── src/
│   ├── components/
│   │   ├── PeopleCounter.jsx   # Main React component
│   │   └── PeopleCounter.css   # Styles & design tokens
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── public/                     # Static assets
├── .env.example                # Env variable docs for mobile builds
├── Dockerfile                  # Multi-stage build (Node → Python)
├── index.html                  # HTML entry point
├── package.json
└── vite.config.js              # Vite config with /api proxy
```

---

## ⚙️ Key Configuration

| Variable | Where | Default | Description |
|----------|-------|---------|-------------|
| `VITE_BACKEND_URL` | `.env` | `http://192.168.1.10:8000` | Backend URL for Capacitor native builds |
| `FRONTEND_ORIGINS` | Server env | `*` | Comma-separated allowed CORS origins for production |
| `CONFIDENCE` | `PeopleCounter.jsx` | `0.40` | Minimum COCO-SSD person confidence |
| `THRESHOLD` | `app.py` | `0.34` | Facenet512 cosine distance match threshold |
| `DETECT_MS` | `PeopleCounter.jsx` | `150` | Detection loop interval in milliseconds |

---

## 🔧 Troubleshooting

| Symptom | Fix |
|---------|-----|
| **Black camera box** | Click "Allow" for camera permissions, close other apps using the webcam (Zoom, Teams, etc.) |
| **Counter not going up** | Check backend terminal for errors. Without backend, app uses fallback `Anon-X` IDs |
| **High CPU usage** | Normal — TensorFlow.js uses WebGL. Close other heavy tabs |
| **Face not recognized across visits** | Hit **Reset Session** — this clears the face database on the server |
| **Docker build fails on `libgl1-mesa-glx`** | Update Dockerfile: use `libgl1` instead (already fixed in this repo) |

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Commit your changes with a clear message
4. Push and open a Pull Request — the CI pipeline will run automatically
5. All checks must pass ✅ before merging

---

## 📄 License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ · <a href="https://aaru07160-vision-count-ai.hf.space/">Live Demo</a> · <a href="https://github.com/Aryan07175/vision-count-ai">GitHub</a>
</div>
