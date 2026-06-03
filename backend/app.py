import sys
import os
import io

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
import time

app = FastAPI()

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for known face embeddings
# Format: { id: [embedding1, embedding2, ...] }
known_faces: Dict[int, List[List[float]]] = {}
next_id = 0

# Cosine distance threshold — Facenet512 produces very tight clusters.
# Same person:      typically 0.00 – 0.25  (even with angle/lighting changes)
# Borderline:       typically 0.25 – 0.35
# Different person: typically 0.40+
# 0.38 = extremely lenient for single-user/small-group counting. 
# Prevents creating new IDs just because the person turned their head.
THRESHOLD = 0.38

# Minimum face area (pixels) — reject tiny/fake face detections
MIN_FACE_AREA = 1500

class ImagePayload(BaseModel):
    image_base64: str

def decode_image(b64_string: str):
    try:
        # Remove data URI header if present
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def cosine_distance(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1 - (dot_product / (norm_a * norm_b))

@app.post("/identify")
async def identify_person(payload: ImagePayload):
    global next_id
    
    img = decode_image(payload.image_base64)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
        
    try:
        # Extract embedding using DeepFace with Facenet512.
        # We try multiple backends for robustness.
        # We use SSD because OpenCV is prone to FALSE POSITIVES (detecting shirts as faces)
        # which causes random new IDs to be generated!
        for backend in ["ssd", "retinaface"]:
            try:
                objs = DeepFace.represent(
                    img_path=img, 
                    model_name="Facenet512", 
                    enforce_detection=True,
                    detector_backend=backend
                )
                if objs and len(objs) > 0:
                    print(f"Face successfully detected using backend: {backend}")
                    break
            except Exception as e:
                err_msg = repr(e).encode('ascii', 'replace').decode('ascii')
                print(f"Backend {backend} failed (no face?): {err_msg}")
                
        if not objs or len(objs) == 0:
            return {"id": None, "status": "no_face", "distance": -1}
            
        face_info = objs[0].get("facial_area", {})
        face_w = face_info.get("w", 0)
        face_h = face_info.get("h", 0)
        face_area = face_w * face_h
        
        # Only reject if the detected face is impossibly small (noise)
        if face_area < 200:
            print(f"Face too small to be real (face_area={face_area})")
            return {"id": None, "status": "no_face", "distance": -1}
            
        embedding = objs[0]["embedding"]
        
        best_match_id = None
        best_distance = float('inf')
        
        # Compare against all known embeddings
        for person_id, encodings in known_faces.items():
            # Use the MINIMUM distance across stored embeddings for robustness.
            # (avg can be pulled up by old/bad embeddings; min finds the best angle match)
            distances = [cosine_distance(embedding, known_emb) for known_emb in encodings]
            min_dist = min(distances)
            if min_dist < best_distance:
                best_distance = min_dist
                best_match_id = person_id
                    
        if best_match_id is not None and best_distance < THRESHOLD:
            print(f"Match found! ID: {best_match_id} (Distance: {best_distance:.3f})")
            # Add this new angle to their profile to improve future matches
            if len(known_faces[best_match_id]) < 8:  # store up to 8 angles
                known_faces[best_match_id].append(embedding)
                
            return {"id": best_match_id, "status": "recognized", "distance": float(best_distance)}
            
        # No match found, create new person
        new_person_id = next_id
        next_id += 1
        known_faces[new_person_id] = [embedding]
        
        print(f"New person identified! Assigned ID: {new_person_id}")
        safe_distance = float(best_distance) if best_distance != float('inf') else -1.0
        return {"id": new_person_id, "status": "new", "distance": safe_distance}
        
    except Exception as e:
        safe_msg = repr(e).encode('ascii', 'replace').decode('ascii')
        print(f"DeepFace error: {safe_msg}")
        # Return no_face instead of 500 so the frontend can retry gracefully
        return {"id": None, "status": "error", "distance": -1}

@app.post("/reset")
async def reset_database():
    global next_id
    known_faces.clear()
    next_id = 0
    print("Database cleared! Starting fresh.")
    return {"status": "reset", "known_people": 0}

@app.get("/status")
async def status():
    return {"status": "running", "known_people": len(known_faces)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
