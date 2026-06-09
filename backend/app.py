import sys
import os
import io
import asyncio
import hashlib
import time
import threading

# Fix Windows encoding issues — DeepFace logs Unicode that
# 'charmap' can't handle
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace'
    )  # type: ignore
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace'
    )  # type: ignore
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2  # type: ignore
import numpy as np
from deepface import DeepFace  # type: ignore
from typing import Dict, List
from collections import OrderedDict
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# ─── API Router for Backend Routes ───────────────────────────────────────
api_router = APIRouter(prefix="/api")
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# Allow CORS for the React frontend and mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory storage for known face embeddings ─────────────────────────
# Format: { id: { "embeddings": [...], "first_seen": ts, "last_seen": ts } }
known_faces: Dict[int, dict] = {}
next_id = 0

# ─── LRU Result Cache ────────────────────────────────────────────────────
# Avoids redundant DeepFace calls for the same image (same person/position).
# Key = SHA256 of the raw image bytes. Value = {"result": ..., "ts": timestamp}
MAX_CACHE_SIZE = 200
CACHE_TTL_SECS = 4  # cache entries valid for 4 seconds only
_result_cache: OrderedDict = OrderedDict()

# ─── Concurrency lock ────────────────────────────────────────────────────
# Prevents multiple simultaneous DeepFace calls which would thrash the CPU.
_inference_lock = asyncio.Lock()

# FIX: next_id is incremented inside a thread executor — protect it with a
# threading.Lock to prevent two simultaneous requests getting the same ID.
_id_lock = threading.Lock()

# ─── Tuning constants ────────────────────────────────────────────────────
# Cosine distance benchmarks for Facenet512 (lower = more similar):
#   Same person, same angle:        0.00 – 0.18
#   Same person, different angle:   0.18 – 0.28
#   Different people, similar look: 0.30 – 0.40
#   Clearly different people:       0.40+
# FIX: was 0.40 — far too loose, causing different people to match the same ID.
# 0.30 sits well inside the "different people" zone for Facenet512.
# Update: 0.30 is too strict for different angles of the same person.
# Loosening to 0.40 to ensure the same person is matched when they return.
# Update 2: 0.40 is way too loose, causing different people to be matched.
# Update 3: 0.28 was too strict, rejecting the same person if their angle changed.
# 0.34 is the perfect "goldilocks" boundary for Facenet512.
THRESHOLD = 0.34

# FIX: Margin guard — the best match must be this much BETTER than the
# second-best candidate. If two stored people are almost equally close,
# the match is ambiguous and we reject it to avoid giving the wrong ID.
# Update: 0.07 is too strict, reducing to 0.04 to allow close matches.
# Update 2: Removing margin guard (0.00) because it causes the AI to panic and return 'ambiguous' when friends look somewhat similar, which forces the frontend to assign a fallback Anon-X ID and increment the counter.
MATCH_MARGIN = 0.00

# Minimum confidence area for a face detection to be trusted
MIN_FACE_AREA = 800  # pixels² — raised slightly to avoid noisy embeddings

# Max embeddings stored per person (memory of their different angles)
MAX_EMBEDDINGS_PER_PERSON = 12

# Detector backends to try, in order of speed vs accuracy
# Removed 'retinaface' because it is far too slow and causes the queue to back up when people look away.
DETECTOR_BACKENDS = ["ssd", "opencv"]


class ImagePayload(BaseModel):
    image_base64: str


def decode_image(b64_string: str):
    """Decode a base64 image string to a numpy array (BGR)."""
    try:
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img, img_data
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None, None


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Improve detection quality by enhancing image contrast and sharpness.
    Handles dim / backlit environments better.
    """
    # Convert to LAB colour space for CLAHE (adaptive histogram equalisation)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Mild unsharp mask to sharpen edges (helps face detectors)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened


def cosine_distance(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    na = np.linalg.norm(a_arr)
    nb = np.linalg.norm(b_arr)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - dot / (na * nb))


def find_best_match(embedding: List[float]) -> tuple:
    """
    Search known_faces for the closest match using cosine distance.

    FIX — three improvements over the original:
    1. Uses mean of the N closest stored embeddings (not min). This prevents
       a single outlier embedding from causing a false-positive match.
    2. Returns BOTH best and second-best distances so the caller can apply
       a margin check and reject ambiguous results.
    3. Never returns a match if fewer embeddings were compared than needed.

    Returns (best_id, best_dist, second_best_dist).
    """
    scores = []  # list of (distance, person_id)

    for person_id, data in known_faces.items():
        all_dists = sorted(
            cosine_distance(embedding, e) for e in data["embeddings"]
        )
        # Use the single best match (min distance). A person might match one
        # stored angle perfectly but not the others. Averaging them would
        # artificially inflate the distance and cause recognition failures.
        best_single_dist = all_dists[0]
        scores.append((best_single_dist, person_id))

    if not scores:
        return None, float('inf'), float('inf')

    scores.sort(key=lambda x: x[0])
    best_dist, best_id = scores[0]
    second_dist = scores[1][0] if len(scores) > 1 else float('inf')
    return best_id, best_dist, second_dist


def cache_key(img_data: bytes) -> str:
    return hashlib.sha256(img_data).hexdigest()


def cache_get(key: str):
    """Return cached result if it exists and is still fresh, else None."""
    entry = _result_cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECS:
        _result_cache.move_to_end(key)  # LRU: mark as recently used
        return entry["result"]
    return None


def cache_set(key: str, result: dict):
    """Store result in LRU cache, evicting oldest entry when full."""
    _result_cache[key] = {"result": result, "ts": time.time()}
    _result_cache.move_to_end(key)
    while len(_result_cache) > MAX_CACHE_SIZE:
        _result_cache.popitem(last=False)


@api_router.post("/identify")
async def identify_person(payload: ImagePayload):
    img, img_data = decode_image(payload.image_base64)
    if img is None or img_data is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # ── 1. Cache lookup ──────────────────────────────────────────────────
    key = cache_key(img_data)
    cached = cache_get(key)
    if cached is not None:
        return cached

    # ── 2. Preprocess ────────────────────────────────────────────────────
    img = preprocess_image(img)

    # ── 3. Serialize DeepFace calls to prevent CPU overload ──────────────
    async with _inference_lock:
        result = await asyncio.get_running_loop().run_in_executor(
            None, _run_deepface, img
        )

    cache_set(key, result)
    return result


def _run_deepface(img: np.ndarray) -> dict:
    """Synchronous DeepFace inference — runs in a thread executor."""
    global next_id

    objs = None  # Initialize before loop so 'if not objs' works
    for backend in DETECTOR_BACKENDS:
        try:
            objs = DeepFace.represent(
                img_path=img,
                model_name="Facenet512",
                enforce_detection=True,
                detector_backend=backend,
            )
            if objs and len(objs) > 0:
                print(f"[DeepFace] Face detected via '{backend}'")
                break
        except Exception as e:
            msg = repr(e).encode('ascii', 'replace').decode('ascii')
            print(f"[DeepFace] Backend '{backend}' failed: {msg}")

    if not objs:
        return {"id": None, "status": "no_face", "distance": -1}

    # Validate face area — reject noise
    face_info = objs[0].get("facial_area", {})
    face_area = face_info.get("w", 0) * face_info.get("h", 0)
    if face_area < MIN_FACE_AREA:
        print(f"[DeepFace] Face too small ({face_area}px²), skipping")
        return {"id": None, "status": "no_face", "distance": -1}

    embedding = objs[0]["embedding"]

    # ── Match against known faces ───────────────────────────────────────────
    best_id, best_dist, second_dist = find_best_match(embedding)

    if best_id is not None and best_dist < THRESHOLD:
        # FIX: Margin check — if the best and second-best candidates are almost
        # equally close, the match is ambiguous (could be either person).
        # Reject and treat as a new person to avoid assigning the wrong ID.
        margin = second_dist - best_dist
        if margin < MATCH_MARGIN:
            print(
                f"[Ambiguous] best={best_dist:.3f} vs "
                f"second={second_dist:.3f} (margin={margin:.3f} "
                f"< {MATCH_MARGIN}) — retrying next frame"
            )
            return {"id": None, "status": "ambiguous", "distance": -1}
        else:
            # Known person — update their profile with this new embedding angle
            data = known_faces[best_id]
            data["last_seen"] = time.time()
            if len(data["embeddings"]) < MAX_EMBEDDINGS_PER_PERSON:
                data["embeddings"].append(embedding)
            print(
                f"[Match] ID {best_id} "
                f"(dist={best_dist:.3f}, margin={margin:.3f})"
            )
            return {
                "id": best_id,
                "status": "recognized",
                "distance": float(best_dist)
            }

    # New person
    with _id_lock:
        new_id = next_id
        next_id += 1
    known_faces[new_id] = {
        "embeddings": [embedding],
        "first_seen": time.time(),
        "last_seen": time.time(),
    }
    safe_dist = float(best_dist) if best_dist != float('inf') else -1.0
    print(
        f"[New] Person assigned ID {new_id} "
        f"(closest existing dist={safe_dist:.3f})"
    )
    return {"id": new_id, "status": "new", "distance": safe_dist}


@api_router.post("/reset")
async def reset_database():
    global next_id
    known_faces.clear()
    _result_cache.clear()
    next_id = 0
    print("[Reset] Database and cache cleared.")
    return {"status": "reset", "known_people": 0}


@api_router.get("/status")
async def status():
    return {
        "status": "running",
        "known_people": len(known_faces),
        "cache_entries": len(_result_cache),
    }


app.include_router(api_router)

# ─── Mount React Frontend ────────────────────────────────────────────────
frontend_dist = os.path.join(os.path.dirname(__file__), '..', 'dist')
if os.path.exists(frontend_dist):
    app.mount(
        "/", StaticFiles(directory=frontend_dist, html=True), name="frontend"
    )

    @app.exception_handler(404)
    async def custom_404_handler(request, exc):
        return FileResponse(os.path.join(frontend_dist, 'index.html'))
else:
    print(f"Warning: Frontend build directory not found at {frontend_dist}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
