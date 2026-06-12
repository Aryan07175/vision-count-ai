# VisionCount AI — Technical Documentation

> **Version:** 2.0 &nbsp;|&nbsp; **Last Updated:** June 2026 &nbsp;|&nbsp; **Live Demo:** [aaru07160-vision-count-ai.hf.space](https://aaru07160-vision-count-ai.hf.space/)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
   - 3.1 [High-Level Overview](#31-high-level-overview)
   - 3.2 [Frontend — Edge Detection Layer](#32-frontend--edge-detection-layer)
   - 3.3 [Backend — Biometric Identity Layer](#33-backend--biometric-identity-layer)
   - 3.4 [Data Flow](#34-data-flow)
4. [Technology Stack](#4-technology-stack)
5. [Key Algorithms](#5-key-algorithms)
   - 5.1 [Non-Maximum Suppression (NMS)](#51-non-maximum-suppression-nms)
   - 5.2 [Centroid Tracking](#52-centroid-tracking)
   - 5.3 [Ghost Memory Registry](#53-ghost-memory-registry)
   - 5.4 [Cosine Distance Matching](#54-cosine-distance-matching)
   - 5.5 [Image Preprocessing (CLAHE)](#55-image-preprocessing-clahe)
   - 5.6 [LRU Result Cache](#56-lru-result-cache)
6. [API Reference](#6-api-reference)
7. [Configuration Reference](#7-configuration-reference)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Docker & Deployment](#9-docker--deployment)
10. [Error Handling & Fallback Strategy](#10-error-handling--fallback-strategy)
11. [Security Considerations](#11-security-considerations)
12. [Changelog](#12-changelog)
13. [Known Limitations](#13-known-limitations)

---

## 1. Executive Summary

VisionCount AI is a full-stack, real-time people counting system that eliminates the most common failure mode of traditional counters: **double counting the same individual**. 

It uses a **two-layer hybrid AI architecture**:
- A lightweight **in-browser model** (TensorFlow.js + COCO-SSD) detects human bodies at ~7 FPS
- A **server-side face recognition model** (Python + DeepFace Facenet512) assigns each person a permanent biometric ID

Even if a person leaves the camera frame and returns hours later, they are matched to their existing ID and counted exactly once.

---

## 2. Problem Statement

Traditional people-counting systems suffer from three critical failure modes:

| Problem | Root Cause | Impact |
|---------|-----------|--------|
| **Double-counting on re-entry** | No persistent person identity — each appearance is treated as new | Count inflated by 2–5× in high-traffic environments |
| **Occlusion failures** | Bounding-box trackers lose identity when people overlap | New IDs assigned to the same person mid-scene |
| **Motion-only counting** | Line-crossing or pixel-diff algorithms count movement, not people | Background movement, animals, or objects trigger false counts |

VisionCount AI addresses all three by combining spatial tracking (short-term memory) with facial biometrics (long-term identity).

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         WEB BROWSER (CLIENT)                          │
│                                                                        │
│  ┌──────────┐    ┌────────────────┐    ┌──────────────────────────┐  │
│  │  Webcam  │───►│  TF.js COCO-   │───►│  NMS + Centroid Tracker  │  │
│  │  Feed    │    │  SSD Detector  │    │  + Ghost Memory Registry  │  │
│  └──────────┘    └────────────────┘    └────────────┬─────────────┘  │
│                                                      │                 │
│                                          New stable  │                 │
│                                          track found │                 │
│                                                      ▼                 │
│                                        ┌─────────────────────────┐   │
│                                        │  Crop face region        │   │
│                                        │  Encode as JPEG base64   │   │
│                                        └──────────┬──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                                     │
                              POST /api/identify      │
                              { image_base64: "..." } │
                                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          FASTAPI BACKEND                               │
│                                                                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│  │  SHA256 LRU  │──►│  DeepFace    │──►│  Cosine Distance Match   │  │
│  │  Cache (4s)  │   │  Facenet512  │   │  vs. known_faces dict    │  │
│  └──────────────┘   └──────────────┘   └────────────┬─────────────┘  │
│                                                      │                 │
│                                         Match < 0.34 │ No match       │
│                                                  ▼   │   ▼            │
│                                          Return   │   │  Assign       │
│                                          existing │   │  new ID       │
│                                          ID       │   │               │
└──────────────────────────────────────────────────────────────────────┘
                                                     │
                              { id, status, distance }│
                                                     ▼
                                        Browser updates count,
                                        logs entry, draws badge
```

### 3.2 Frontend — Edge Detection Layer

The browser acts as a real-time edge compute node. All person body detection and spatial tracking happens locally — no raw video is ever sent to the server.

**Detection loop** (`DETECT_MS = 150ms`):

1. Grab a video frame from the webcam via the `<video>` element
2. Run `cocoSsd.detect(videoElement)` — returns bounding boxes for all detected objects
3. Filter to `class === "person"` with `score ≥ 0.40`
4. Apply Non-Maximum Suppression to remove duplicate boxes
5. Convert boxes to centroid coordinates `(cx, cy)`
6. Run greedy nearest-neighbour matching against existing tracks
7. Update track states: `appeared`, `disappeared`, `counted`, `identifying`
8. For stable, uncounted tracks (`appeared ≥ 5`): asynchronously call `/api/identify`
9. Draw overlay on `<canvas>`: corner brackets, scan bar, ID badge, ghost rings

**Track States:**

```
NEW → SCANNING (appeared 1–4) → IDENTIFYING (calling API) → COUNTED (ID assigned)
                                       ↓ (backend offline or timeout)
                                  ANON-X (anonymous fallback count)
```

### 3.3 Backend — Biometric Identity Layer

The FastAPI server receives cropped face images, extracts biometric embeddings, and matches them against an in-memory database.

**Request pipeline per `/api/identify` call:**

```
1. Decode base64 JPEG → NumPy array
2. SHA256 hash → check LRU cache (4s TTL, 200 entries max)
   └─ Cache HIT  → return cached result immediately
   └─ Cache MISS → continue
3. Acquire asyncio.Lock (serialize inference — prevent CPU overload)
4. CLAHE preprocessing (contrast + sharpness enhancement)
5. DeepFace.represent() with SSD detector → fallback to OpenCV
6. Validate face area ≥ 800 px²
7. Extract 512-dim Facenet512 embedding
8. Cosine distance vs. all stored embeddings
9. Best match < 0.34 threshold → return existing ID + update embedding store
   No match → assign next sequential ID → store embedding
10. Cache result → release lock → return JSON
```

**In-memory storage format:**
```python
known_faces: Dict[int, {
    "embeddings": List[List[float]],  # up to 12 angle variants
    "first_seen": float,              # Unix timestamp
    "last_seen":  float
}]
```

### 3.4 Data Flow

```
Timeline for a single person being counted:

Frame 1:   COCO-SSD detects person → track "tmp_0" created (appeared=1)
Frame 2–4: Track updated, appeared increases
Frame 5:   appeared ≥ MIN_FRAMES → identifyPerson() called asynchronously
           (track marked identifying=true to prevent duplicate requests)
           
           ┌── API response arrives ──────────────────────┐
           │  Case A: { id: 3, status: "new" }            │
           │    → personId=3, counted=true                 │
           │    → totalCount++, log entry added            │
           │                                               │
           │  Case B: { id: 3, status: "recognized" }     │
           │    → Already in countedIds set → no increment │
           │    → badge shows "↩ RETURNING (ID 3)"        │
           │                                               │
           │  Case C: { id: null, status: "no_face" }     │
           │    → identifying=false, retry next frame      │
           │    → After 30 frames (~4.5s): Anon-X fallback │
           │                                               │
           │  Case D: network error                        │
           │    → Anon-X assigned immediately              │
           │    → apiError banner shown                    │
           └──────────────────────────────────────────────┘

Person leaves frame:
  → disappeared counter increments each frame
  → After MAX_DISAPPEARED (10) frames: moved to ghost registry
  
Person returns:
  → Ghost registry matched by proximity (GHOST_RATIO of diagonal)
  → Track revived with original ID and counted=true
  → No new API call, no count increment
```

---

## 4. Technology Stack

### Frontend

| Package | Version | Purpose |
|---------|---------|---------|
| `react` | ^19.2.6 | UI component framework |
| `react-dom` | ^19.2.6 | DOM rendering |
| `react-webcam` | ^7.2.0 | Webcam stream access |
| `@tensorflow/tfjs` | ^4.22.0 | In-browser ML runtime (WebGL backend) |
| `@tensorflow-models/coco-ssd` | ^2.2.3 | Pre-trained MobileNetV2 object detector |
| `@capacitor/core` | ^6.0.0 | Native mobile wrapper (iOS/Android) |
| `vite` | ^8.0.12 | Build tool and dev server |
| `@vitejs/plugin-react` | ^6.0.1 | React fast-refresh + JSX transform |
| `eslint` | ^10.3.0 | Code linting |

### Backend

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.115.0,<1.0 | REST API framework |
| `uvicorn[standard]` | ≥0.32.0,<1.0 | ASGI server |
| `pydantic` | ≥2.0.0,<3.0 | Request/response validation |
| `opencv-python-headless` | ≥4.8.0,<5.0 | Image processing (no GUI) |
| `numpy` | ≥1.24.0,<3.0 | Numerical operations |
| `deepface` | ≥0.0.93 | Face recognition framework |
| `tf-keras` | ≥2.16.0 | Required by DeepFace models |

### Infrastructure

| Tool | Purpose |
|------|---------|
| **Docker** (multi-stage) | Single image: Node build → Python runtime |
| **Hugging Face Spaces** | Cloud hosting, Docker SDK, port 7860 |
| **GitHub Actions** | CI pipeline — lint, build, syntax check |
| **Vite proxy** | Routes `/api/*` to `localhost:8000` in dev |

---

## 5. Key Algorithms

### 5.1 Non-Maximum Suppression (NMS)

Eliminates overlapping duplicate detections of the same person.

```
Input:  All "person" bounding boxes sorted by confidence score (high → low)
Output: Filtered list — at most one box per body

For each box (high confidence first):
  Compute IoU (Intersection over Union) and IoM (Intersection over Min)
  vs. every already-kept box.
  Suppress if:  IoU > 0.50  (boxes overlap by >50% of combined area)
             OR IoM > 0.85  (one box is nearly contained in the other)
  Otherwise: keep it.

Constants: NMS_IOU=0.50, NMS_IOM=0.85
```

### 5.2 Centroid Tracking

Matches detected bodies frame-to-frame using greedy nearest-neighbour assignment.

```
For each existing track and each new detection:
  Compute Euclidean distance between centroids.
  Threshold: diagLen × MATCH_RATIO (8% of frame diagonal)

Sort all (track, detection) candidate pairs by distance.
Greedily assign: each track and detection used at most once.

Result:
  matched        → [(trackId, detection), ...]
  unmatchedTracks → tracks that had no detection this frame
  unmatchedDets   → detections that had no matching track (new people)
```

### 5.3 Ghost Memory Registry

Prevents a momentary disappearance (occlusion, frame edge) from creating a new person.

```
When a track disappears for MAX_DISAPPEARED (10) frames:
  Move from active → ghost registry with last known (cx, cy, frame)

On each new detection that fails active track matching:
  Check all ghosts:
    if euclidean(ghost.cx, ghost.cy, det.cx, det.cy) < diagLen × GHOST_RATIO (5%)
      Revive ghost as active track with original ID and counted=true
      Ghost TTL: GHOST_TTL=3 frames after being moved to ghost registry
```

### 5.4 Cosine Distance Matching

Used by the backend to compare facial embeddings.

```
distance = 1 - (a · b) / (‖a‖ × ‖b‖)

Range: 0.0 (identical) → 2.0 (opposite)

Empirical thresholds for Facenet512:
  0.00–0.18  Same person, same angle
  0.18–0.28  Same person, different angle
  0.28–0.34  Borderline (used as threshold)
  0.34–0.40  Likely different people
  0.40+      Clearly different people

THRESHOLD = 0.34
```

For each incoming embedding, the system compares against the closest stored embedding for each known person (not the average — this handles different viewing angles better).

### 5.5 Image Preprocessing (CLAHE)

Applied to every image before DeepFace inference to improve detection in dim/backlit conditions.

```
1. Convert BGR → LAB colour space
2. Apply CLAHE to L channel:
   clipLimit=2.0, tileGridSize=(8,8)
   (Contrast Limited Adaptive Histogram Equalisation)
3. Merge enhanced L with original A, B
4. Convert LAB → BGR
5. Apply unsharp masking:
   blurred   = GaussianBlur(enhanced, sigma=3)
   sharpened = enhanced × 1.5 + blurred × (-0.5)
```

### 5.6 LRU Result Cache

Prevents redundant DeepFace inference for the same image (same person, same position).

```
Key:   SHA256(raw JPEG bytes) — identical images produce identical keys
Value: { result: dict, ts: float }
TTL:   4 seconds — stale entries are ignored
Size:  200 entries max — LRU eviction when full

Cache HIT:  result returned in <1ms (vs ~500ms inference)
Cache MISS: full inference pipeline runs, result stored
```

---

## 6. API Reference

Base URL (development): `http://localhost:8000`  
Base URL (production): served under the same origin as the frontend

### `POST /api/identify`

Identifies a person from a cropped face image.

**Request body:**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

The `data:image/...;base64,` prefix is automatically stripped if present.

**Response — new person:**
```json
{
  "id": 0,
  "status": "new",
  "distance": -1.0
}
```

**Response — recognized person:**
```json
{
  "id": 3,
  "status": "recognized",
  "distance": 0.218
}
```

**Response — no face detected / too small:**
```json
{
  "id": null,
  "status": "no_face",
  "distance": -1
}
```

**Response — ambiguous match (margin guard):**
```json
{
  "id": null,
  "status": "ambiguous",
  "distance": -1
}
```

**Error responses:**
| Code | Meaning |
|------|---------|
| `400` | Invalid or undecodable image data |
| `500` | Internal server error |

---

### `POST /api/reset`

Clears the face database and result cache. Used when starting a new counting session.

**Response:**
```json
{
  "status": "reset",
  "known_people": 0
}
```

---

### `GET /api/status`

Health check — returns current database state.

**Response:**
```json
{
  "status": "running",
  "known_people": 12,
  "cache_entries": 45
}
```

---

## 7. Configuration Reference

### Frontend Constants (`src/components/PeopleCounter.jsx`)

| Constant | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE` | `0.40` | Min COCO-SSD person confidence score (0–1) |
| `NMS_IOU` | `0.50` | Max IoU before a box is suppressed |
| `NMS_IOM` | `0.85` | Max IoM before a box is suppressed |
| `MIN_FRAMES` | `5` | Frames a person must be stable before identification |
| `MAX_DISAPPEARED` | `10` | Frames before a track is moved to ghost registry |
| `GHOST_TTL` | `3` | Frames a ghost entry survives |
| `MATCH_RATIO` | `0.08` | Max centroid distance as fraction of frame diagonal |
| `GHOST_RATIO` | `0.05` | Max ghost revival distance as fraction of frame diagonal |
| `DUPE_RATIO` | `0.05` | Min distance between tracks to not be considered duplicate |
| `DETECT_MS` | `150` | Detection loop interval in milliseconds (~6.7 FPS) |
| `IDENTIFY_TIMEOUT` | `10000` | Max ms to wait for backend response before abort |

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_BACKEND_URL` | `http://192.168.1.10:8000` | Backend URL for Capacitor native builds only |

### Backend Constants (`backend/app.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `THRESHOLD` | `0.34` | Facenet512 cosine distance match threshold |
| `MATCH_MARGIN` | `0.00` | Min distance gap between 1st and 2nd best match (0 = disabled) |
| `MIN_FACE_AREA` | `800` | Minimum face bounding box area in pixels² |
| `MAX_EMBEDDINGS_PER_PERSON` | `12` | Max stored embedding angles per person |
| `MAX_CACHE_SIZE` | `200` | LRU cache maximum entries |
| `CACHE_TTL_SECS` | `4` | LRU cache entry lifetime in seconds |
| `DETECTOR_BACKENDS` | `["ssd", "opencv"]` | Face detector order (fastest first) |

### Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRONTEND_ORIGINS` | `*` | Comma-separated allowed CORS origins. Use `*` for public APIs, restrict in production |

---

## 8. CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci.yml`)

Runs automatically on every push to `main`, `fix/**`, `feat/**`, `chore/**` branches and all pull requests targeting `main`.

```
Push / PR
    │
    ▼
┌──────────────────────────────────┐
│  Job 1: Frontend (runs parallel) │
│  ─────────────────────────────── │
│  1. actions/checkout@v4          │
│  2. actions/setup-node@v4        │
│     Node 20.x, npm cache         │
│  3. npm ci                       │
│  4. npm run lint (ESLint)        │
│  5. npm run build (Vite)         │
│  6. Upload dist/ artifact (7d)   │
└──────────────────────────────────┘
┌──────────────────────────────────┐
│  Job 2: Backend (runs parallel)  │
│  ─────────────────────────────── │
│  1. actions/checkout@v4          │
│  2. actions/setup-python@v5      │
│     Python 3.10, pip cache       │
│  3. pip install -r requirements  │
│  4. python -m py_compile app.py  │
│  5. FastAPI import check         │
└──────────────────────────────────┘
    │
    │ All checks pass ✅
    ▼
  PR can be merged → main
    │
    ▼
Hugging Face Spaces detects push
    │
    ▼
Docker build (multi-stage):
  Stage 1: node:20-alpine → npm build → /app/dist
  Stage 2: python:3.10-slim → pip install → copy dist → uvicorn
    │
    ▼
Container live at https://aaru07160-vision-count-ai.hf.space/
```

### Branch Strategy

| Branch Pattern | Purpose | CI Runs |
|----------------|---------|---------|
| `main` | Production (auto-deploys to HF Spaces) | ✅ Full CI |
| `fix/**` | Bug fix PRs | ✅ Full CI |
| `feat/**` | New feature PRs | ✅ Full CI |
| `chore/**` | Maintenance, docs, dependency updates | ✅ Full CI |

---

## 9. Docker & Deployment

### Multi-Stage Dockerfile

```dockerfile
# Stage 1 — Build React frontend
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build            # outputs → /app/dist

# Stage 2 — Python runtime + serve everything
FROM python:3.10-slim

# System libs for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces: must run as non-root UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Python deps
COPY --chown=user backend/requirements.txt ./backend/
RUN pip install --user --no-cache-dir -r backend/requirements.txt

# Frontend build + backend source
COPY --chown=user --from=build /app/dist ./dist
COPY --chown=user backend/ ./backend/

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Why multi-stage?**
- Stage 1 uses the full Node.js toolchain to build the frontend. The resulting `dist/` folder is just static HTML/CSS/JS.
- Stage 2 only contains the Python runtime and the compiled frontend. No Node.js, no dev dependencies. Image stays lean.

**Why `libgl1` not `libgl1-mesa-glx`?**  
`libgl1-mesa-glx` was removed in Debian Trixie (which `python:3.10-slim` now uses). The replacement is `libgl1`. This was the root cause of an earlier build failure documented in `hf_build_error.txt`.

### Hugging Face Spaces Configuration

The YAML front-matter in `README.md` configures the Space:

```yaml
title: Vision Count AI
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
```

The `app_port: 7860` tells HF Spaces to expose port 7860, which is what Uvicorn listens on.

---

## 10. Error Handling & Fallback Strategy

The system is designed to **always count people**, even when components fail.

```
Backend available + face detected
    └─► Full face ID assigned (ID 0, 1, 2, ...)
    
Backend available + no face detected (obscured / too small)
    └─► Retry for up to ~4.5 s (30 frames × 150ms)
    └─► If still no face: Anon-X fallback count
    
Backend offline / network error
    └─► apiError banner shown to user
    └─► Immediate Anon-X fallback count
    └─► App continues working fully in offline mode
    
Backend request timeout (> 10 s)
    └─► AbortController fires, fetch cancelled
    └─► Same Anon-X fallback path
    
Identification stuck (> 10 s — rare race condition)
    └─► IDENTIFY_TIMEOUT guard releases the lock
    └─► Track re-enters identification on next frame
    
Webcam permission denied
    └─► Camera error state shown with instructions
    └─► No crash — app renders error UI cleanly
    
Global JS exception / unhandled Promise
    └─► Caught by window.onerror and unhandledrejection handlers
    └─► Shown as dismissable toast banner
    └─► Does not crash the detection loop
```

---

## 11. Security Considerations

| Area | Implementation | Notes |
|------|---------------|-------|
| **CORS** | `FRONTEND_ORIGINS` env var controls allowed origins | Default `*` for public demo; restrict in private deployments |
| **Face data** | Stored in-memory only — wiped on server restart or `/api/reset` | No persistent database, no disk writes |
| **Image transmission** | JPEG crop of face region only — no raw video stream | Minimum data transfer |
| **Credentials** | `allow_credentials=False` in CORS | No cookies or auth headers |
| **Container** | Runs as non-root UID 1000 | Required by Hugging Face Spaces; also best practice |
| **Rate limiting** | `asyncio.Lock` serializes inference | Prevents server overload; one request processed at a time |
| **Input validation** | Pydantic `ImagePayload` model validates request schema | Rejects malformed JSON at framework level |

**Privacy note:** Face embeddings (512-dimensional float vectors) are stored in RAM, not images. The original face image is never persisted. All data is lost when the server process restarts.

---

## 12. Changelog

### v2.0 — June 2026 (current branch: `fix/bugs-and-ui-improvements`)

**Bug Fixes:**
- `FRONTEND_ORIGINS` environment variable was parsed but never used in CORS middleware (was hardcoded to `["*"]`). Now properly wired.
- `window.countAnimTimeout` used as a global timer store — bypassed React lifecycle, leaked on unmount. Replaced with `useRef`.
- `unhandledrejection` event listener added in `useEffect` but never removed in cleanup — leaked on every component remount. Fixed.
- `globalError` state was populated by error handlers but never rendered. Now shown as a dismissable toast banner.
- Native mobile backend URL was hardcoded (`192.168.1.10:8000`). Now reads from `import.meta.env.VITE_BACKEND_URL`.
- `@capacitor/core` was imported in JSX but missing from `dependencies` (only `@capacitor/cli` was listed as devDependency). Production builds would fail to resolve it.
- Dead `useMemo` for `videoConstraints` — the memoised object was never passed to `<Webcam>` component. Removed.
- "How It Works" panel showed "≥ 50% confidence" but actual constant is `0.40` (40%). Corrected.

**Code Quality:**
- `IDENTIFY_TIMEOUT` constant now used consistently (was hardcoded as `10000` a second time inside `fetch()`).
- Dedicated `useEffect` cleanup for `countAnimTimeout` on component unmount.
- Reset now also clears `apiError` and `globalError` states.
- Added `String(id)` cast before `.replace()` to prevent `TypeError` when `id` is numeric.

**Visual / UX:**
- Animated dot-grid background (subtle tech-aesthetic).
- Floating logo bob animation.
- Live session timer (HH:MM:SS) in header — resets on "Reset Session".
- Pulsing ring aura behind the count number.
- Corner-bracket bounding boxes instead of plain rectangles.
- Card shimmer highlight on hover via `::after` pseudo-element.
- Camera frame teal glow on hover.
- Camera flip button rotates 30° on hover.
- Improved spinner with glow shadow.
- API error banner redesigned with icon and backdrop blur.
- Global error shown as dismissable slide-in toast.

**Documentation & CI:**
- README completely rewritten with architecture diagrams, tech stack, project structure, config table, and contributing guide.
- Added GitHub Actions CI pipeline (`.github/workflows/ci.yml`): frontend lint + build, backend syntax + import check.
- Added `.env.example` documenting `VITE_BACKEND_URL`.

---

### v1.x — May–June 2026

**v1.5 — Mirroring, Aspect Ratio & Crop Fixes:**
- Canvas overlay coordinates now flip horizontally in mirrored camera mode (`facingMode === 'user'`).
- Added `object-fit: cover` to canvas element to match webcam scaling in 16:9 container.
- Face crop coordinates clamped within video dimensions to prevent rendering errors.

**v1.4 — Capacitor Dependencies:**
- Added `@capacitor/cli` for native mobile build tooling.

**v1.3 — Performance & Stability:**
- Resolved WebGL context crash on heavy load by fixing webcam resolution to 640×480.
- Extended API timeout from 6s to 15s for CPU-constrained environments.

**v1.2 — Unified Deployment:**
- Multi-stage Dockerfile: React build in Node → served by FastAPI static mount.
- Eliminated CORS complexity by serving frontend and backend from same origin.

**v1.1 — Linux Headless Compatibility:**
- Switched to `opencv-python-headless` (no GUI dependencies).
- Added `libgl1` and `libglib2.0-0` to Dockerfile (fixes OpenCV import on Debian slim).
- Fixed `libgl1-mesa-glx` → `libgl1` (package removed in Debian Trixie).

**v1.0 — Initial Release:**
- TensorFlow.js COCO-SSD body detection.
- DeepFace Facenet512 face identity backend.
- Centroid tracker with ghost memory.
- Hugging Face Spaces deployment.

---

## 13. Known Limitations

| Limitation | Detail | Potential Improvement |
|-----------|--------|----------------------|
| **Server-side face DB is ephemeral** | Known faces are lost on every server restart | Add SQLite or Redis persistence layer |
| **Single-process inference** | `asyncio.Lock` serializes all requests — one at a time | Worker pool or GPU inference service |
| **No authentication** | The `/api/reset` endpoint is publicly accessible | Add API key middleware in production |
| **No face DB size limit** | `known_faces` dict grows unbounded over long sessions | Add LRU eviction or max-age on entries |
| **Face recognition requires front-facing** | Heavily occluded or profile faces may not match | Add multi-angle embedding strategy |
| **Mobile native URL hardcoded** | Falls back to `192.168.1.10:8000` if env not set | Document and enforce `.env` for native builds |
| **TF.js model download on first load** | COCO-SSD (~20 MB) downloads on first visit | Cache with Service Worker |
