import sys
import os
import io
import asyncio
import hashlib
import time

# Fix Windows encoding issues — DeepFace logs Unicode that 'charmap' can't handle
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from deepface import DeepFace
from typing import Dict, List, Optional
from collections import OrderedDict

app = FastAPI()

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory storage for known face embeddings ───────────────────────────────
# Format: { id: { "embeddings": [...], "first_seen": timestamp, "last_seen": timestamp } }
known_faces: Dict[int, dict] = {}
next_id = 0

# ─── LRU Result Cache ──────────────────────────────────────────────────────────
# Avoids redundant DeepFace calls for the same image (same person in same position).
# Key = SHA256 of the raw image bytes. Value = {"result": ..., "ts": timestamp}
MAX_CACHE_SIZE = 200
CACHE_TTL_SECS = 4  # cache entries valid for 4 seconds only (live video changes fast)
_result_cache: OrderedDict = OrderedDict()

# ─── Concurrency lock ──────────────────────────────────────────────────────────
# Prevents multiple simultaneous DeepFace calls which would thrash the CPU.
_inference_lock = asyncio.Lock()

# ─── Tuning constants ──────────────────────────────────────────────────────────
# Cosine distance thresholds (lower = more strict match):
#   Same person (good angle):     ~0.00 – 0.22
#   Same person (angle change):   ~0.22 – 0.35
#   Different people:             ~0.40+
THRESHOLD = 0.40

# Minimum confidence area for a face detection to be trusted
MIN_FACE_AREA = 600  # pixels^2 — lower than before to catch smaller/farther faces

# Max embeddings stored per person (memory of their different angles)
MAX_EMBEDDINGS_PER_PERSON = 12

# Detector backends to try, in order of speed vs accuracy
DETECTOR_BACKENDS = ["ssd", "opencv", "retinaface"]


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
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
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
    Search known_faces for the closest match using minimum cosine distance.
    Returns (best_id, best_distance) or (None, inf).
    """
    best_id = None
    best_dist = float('inf')
    for person_id, data in known_faces.items():
        dists = [cosine_distance(embedding, e) for e in data["embeddings"]]
        d = min(dists)
        if d < best_dist:
            best_dist = d
            best_id = person_id
    return best_id, best_dist


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


@app.post("/identify")
async def identify_person(payload: ImagePayload):
    global next_id

    img, img_data = decode_image(payload.image_base64)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # ── 1. Cache lookup ────────────────────────────────────────────────────────
    key = cache_key(img_data)
    cached = cache_get(key)
    if cached is not None:
        return cached

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    img = preprocess_image(img)

    # ── 3. Serialize DeepFace calls to prevent CPU overload ───────────────────
    async with _inference_lock:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_deepface, img
        )

    cache_set(key, result)
    return result


def _run_deepface(img: np.ndarray) -> dict:
    """Synchronous DeepFace inference — runs in a thread executor."""
    global next_id

    objs = None
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
    best_id, best_dist = find_best_match(embedding)

    if best_id is not None and best_dist < THRESHOLD:
        # Known person — update their profile
        data = known_faces[best_id]
        data["last_seen"] = time.time()
        if len(data["embeddings"]) < MAX_EMBEDDINGS_PER_PERSON:
            data["embeddings"].append(embedding)
        print(f"[Match] ID {best_id} (dist={best_dist:.3f})")
        return {"id": best_id, "status": "recognized", "distance": float(best_dist)}

    # New person
    new_id = next_id
    next_id += 1
    known_faces[new_id] = {
        "embeddings": [embedding],
        "first_seen": time.time(),
        "last_seen": time.time(),
    }
    safe_dist = float(best_dist) if best_dist != float('inf') else -1.0
    print(f"[New] Person assigned ID {new_id}")
    return {"id": new_id, "status": "new", "distance": safe_dist}


@app.post("/reset")
async def reset_database():
    global next_id
    known_faces.clear()
    _result_cache.clear()
    next_id = 0
    print("[Reset] Database and cache cleared.")
    return {"status": "reset", "known_people": 0}


@app.get("/status")
async def status():
    return {
        "status": "running",
        "known_people": len(known_faces),
        "cache_entries": len(_result_cache),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
