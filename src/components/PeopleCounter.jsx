import { useEffect, useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
// TF imports are loaded dynamically inside the model-loading useEffect.
// This removes them from the critical-path bundle (saves ~1.3 MB on first load).
import './PeopleCounter.css';

// ─── CONSTANTS ─────────────────────────────────────────────────────────────────
const CONFIDENCE      = 0.35;   // low threshold → catch far/partially visible people
const NMS_IOU         = 0.60;   // relaxed → separate people standing close aren't merged
const NMS_IOM         = 0.85;   // suppress near-duplicate boxes
const MIN_FRAMES      = 4;      // frames stable before identification attempt
const MAX_DISAPPEARED = 30;     // ~4.5s before moving to ghost (handles occlusions)
const GHOST_TTL_MS    = 3000;   // 3 real seconds — only meant for brief occlusions, not minutes
const MATCH_RATIO     = 0.12;   // ~175px max movement per 150ms frame
const GHOST_RATIO     = 0.08;   // ~115px max distance to revive a ghost
const DUPE_RATIO      = 0.03;
const MAX_DETECTIONS  = 50;     // tell COCO-SSD to return up to 50 people
const DETECT_MS       = 150;
const IDENTIFY_TIMEOUT = 15000; // 15s max wait for DeepFace (prevents 'backend offline' on CPU)
const MAX_NO_FACE_ATTEMPTS = 3; // after 3 failed attempts, count locally (prevents instant overcounting)

// ─── HELPERS ───────────────────────────────────────────────────────────────────
const euclidean = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);

function boxOverlap(a, b) {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const ix1 = Math.max(ax, bx), iy1 = Math.max(ay, by);
  const ix2 = Math.min(ax + aw, bx + bw), iy2 = Math.min(ay + ah, by + bh);
  if (ix2 <= ix1 || iy2 <= iy1) return { iou: 0, iom: 0 };
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const aA = aw * ah, aB = bw * bh;
  return {
    iou: inter / (aA + aB - inter),
    iom: inter / Math.min(aA, aB),
  };
}

// Removes duplicate/nested boxes for the same body
function applyNMS(predictions) {
  const sorted = [...predictions].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const p of sorted) {
    const suppressed = kept.some(k => {
      const { iou, iom } = boxOverlap(p.bbox, k.bbox);
      return iou > NMS_IOU || iom > NMS_IOM;
    });
    if (!suppressed) kept.push(p);
  }
  return kept;
}

// Greedy nearest-neighbour assignment. Returns actual det objects (not indices).
function greedyMatch(tracks, dets, maxDist) {
  const candidates = [];
  for (const [id, track] of tracks) {
    for (let di = 0; di < dets.length; di++) {
      const d = euclidean(track.cx, track.cy, dets[di].cx, dets[di].cy);
      if (d <= maxDist) candidates.push({ id, di, d });
    }
  }
  candidates.sort((a, b) => a.d - b.d);

  const usedIds = new Set();
  const usedDi  = new Set();
  const matched = [];                // [[id, detObject], ...]

  for (const { id, di } of candidates) {
    if (usedIds.has(id) || usedDi.has(di)) continue;
    matched.push([id, dets[di]]);
    usedIds.add(id);
    usedDi.add(di);
  }

  const unmatchedTracks = [...tracks.entries()].filter(([id]) => !usedIds.has(id));
  const unmatchedDets   = dets.filter((_, di) => !usedDi.has(di));
  return { matched, unmatchedTracks, unmatchedDets };
}

function drawBadge(ctx, x, y, label, color) {
  ctx.save();
  ctx.font = 'bold 12px Inter, system-ui, monospace';
  const w = ctx.measureText(label).width + 16;
  ctx.fillStyle = color;
  // rounded rect
  const r = 5, h = 22;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#000';
  ctx.fillText(label, x + 8, y + 15);
  ctx.restore();
}

// ─── API HELPER ───────────────────────────────────────────────────────────────
// BUG FIX: When the webcam is mirrored (selfie mode), the video stream pixels
// are NOT flipped — only the CSS display is. So we must un-mirror the canvas
// crop before sending to backend. We pass `isMirrored` to handle this.
async function identifyPerson(video, box, isMirrored) {
  const [x, y, w, h] = box;
  const vW = video.videoWidth;
  
  // BUG FIX: Send the entire bounding box instead of just the top 35%. 
  // DeepFace's built-in face detector is much better at finding the face 
  // within the full body crop, whereas 35% might chop off the face if seated!
  const cropW = w;
  const cropH = h;
  // BUG FIX: If camera is mirrored (front-facing), the CSS flips it but the
  // raw video stream coordinates are the mirror of what appears on screen.
  // We must flip x-coordinate to match real pixel position in the stream.
  const realX = isMirrored ? vW - x - w : x;
  const cropX = realX;
  const cropY = y;

  const canvas = document.createElement('canvas');
  canvas.width = cropW;
  canvas.height = cropH;
  const ctx = canvas.getContext('2d');
  
  // Draw the video offset to perfectly crop the head/face area
  ctx.drawImage(video, -cropX, -cropY, video.videoWidth, video.videoHeight);
  const base64 = canvas.toDataURL('image/jpeg', 0.8);
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), IDENTIFY_TIMEOUT);
    
    const res = await fetch('/api/identify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: base64 }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error('API Error');
    return await res.json();
  } catch (err) {
    if (err.name === 'AbortError') {
      console.warn("DeepFace API timeout after", IDENTIFY_TIMEOUT, "ms");
    } else {
      console.warn("DeepFace API error:", err);
    }
    // BUG FIX: Return a special marker so caller knows backend is offline
    // and can fall back to local-only counting
    return { id: null, status: 'offline', distance: -1 };
  }
}

// ─── MODULE-LEVEL SINGLETONS ──────────────────────────────────────────────────
// FIX (Issue #5): React 18 StrictMode deliberately unmounts and remounts every
// component once in development. If tracking state lives inside useRef() it gets
// silently reset on the second mount, losing any detections made in the first
// mount cycle. By hoisting these to module scope they survive the remount.
// There is only ever one PeopleCounter on screen, so a singleton is safe.
const _active      = new Map();   // active tracks
const _ghost       = new Map();   // ghost registry
const _countedIds  = new Set();   // permanent counted-ID ledger
let   _nextId      = 0;
let   _localNextId = 10000;       // offset from backend IDs (which start at 0)
let   _loadPromise = null;        // shared TF model promise

// ─── COMPONENT ────────────────────────────────────────────────────────────────
export default function PeopleCounter() {
  const webcamRef   = useRef(null);
  const canvasRef   = useRef(null);
  const rafRef      = useRef(null);
  const lastTickRef = useRef(0);
  const isDetectingRef = useRef(false); // Prevents overlapping inference calls
  const modelRef    = useRef(null);

  // FIX (Issue #5): Point all refs at the module-level singletons so their
  // contents persist across StrictMode remount cycles in development.
  const activeRef          = useRef(_active);
  const ghostRef           = useRef(_ghost);
  const countedIds         = useRef(_countedIds);
  const nextId             = useRef(_nextId);

  // BUG FIX: Local fallback counter starts at 10000 to NEVER clash with
  // backend Face-IDs (which start at 0 and count up: 0, 1, 2...).
  // Without this offset, Person A (Face-ID 0) and Person B (Local-ID 0)
  // would show the same ID number in the UI.
  const localNextPersonId  = useRef(_localNextId);
  // FIX: Use a ref instead of window.countAnimTimeout to avoid global
  // namespace pollution and setState-on-unmounted-component warnings.
  const countAnimRef = useRef(null);

  const [totalCount,  setTotalCount]  = useState(0);
  const [justCounted, setJustCounted] = useState(false);
  const [modelReady,  setModelReady]  = useState(false);
  const [camError,    setCamError]    = useState(false);
  const [facing,      setFacing]      = useState('user');
  const [inFrame,     setInFrame]     = useState(0);
  const [ghosts,      setGhosts]      = useState(0);
  const [log,         setLog]         = useState([]);
  const [backendOnline, setBackendOnline] = useState(true);
  const [sessionTime,   setSessionTime]   = useState(0); // seconds elapsed

  // ── Session timer ─────────────────────────────────────────────────────────
  useEffect(() => {
    const timer = setInterval(() => setSessionTime(s => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const fmtTime = s => {
    const h = String(Math.floor(s / 3600)).padStart(2, '0');
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const sec = String(s % 60).padStart(2, '0');
    return `${h}:${m}:${sec}`;
  };

  // ── Load model ──────────────────────────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    (async () => {
      // FIX (Issue #5): Use the module-level _loadPromise so the expensive TF
      // download is shared across StrictMode remount cycles — never re-fetched.
      if (!_loadPromise) {
        _loadPromise = (async () => {
          // Dynamic imports keep TF.js & COCO-SSD out of the initial JS bundle,
          // cutting first-load payload from ~1.3 MB to a few KB.
          const tf      = await import('@tensorflow/tfjs');
          const cocoSsd = await import('@tensorflow-models/coco-ssd');
          // FIX: Explicitly select a backend so we don't silently fall back to
          // CPU. WebGL gives 10-20× faster inference on most devices.
          try {
            await tf.setBackend('webgl');
          } catch {
            console.warn('[TF] WebGL unavailable, falling back to CPU backend');
            await tf.setBackend('cpu');
          }
          await tf.ready();
          return await cocoSsd.load({ base: 'mobilenet_v2' });
        })();
      }
      
      try {
        const model = await _loadPromise;
        if (!alive) return;
        modelRef.current = model;
        setModelReady(true);
      } catch (err) {
        console.error("Error loading model:", err);
      }
    })();
    return () => {
      alive = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // ── Detection loop ──────────────────────────────────────────────────────────
  const detect = useCallback(async () => {
    if (isDetectingRef.current) return;
    
    const model  = modelRef.current;
    const webcam = webcamRef.current;
    const canvas = canvasRef.current;
    if (!model || !webcam || !canvas) return;
    const video = webcam.video;
    if (!video || video.readyState !== 4) return;
    const vW = video.videoWidth, vH = video.videoHeight;
    if (!vW || !vH) return;

    isDetectingRef.current = true;

    try {
      // BUG FIX: Track whether webcam is currently mirrored for coordinate correction
    const isMirrored = facing === 'user';

    canvas.width = vW; canvas.height = vH;
    const diagLen   = Math.hypot(vW, vH);
    const ghostDist = diagLen * GHOST_RATIO;

    // ── 1. Detect persons only ─────────────────────────────────────────────
    // Pass MAX_DETECTIONS so COCO-SSD returns up to 30 boxes (default is only 20)
    const raw = await model.detect(video, MAX_DETECTIONS);
    const dets = applyNMS(
      raw.filter(
        p =>
          typeof p.class === 'string' &&
          p.class.trim().toLowerCase() === 'person' &&
          p.score >= CONFIDENCE &&
          p.bbox[2] > 15 &&   // allow narrower boxes (partially visible)
          p.bbox[3] > 30      // allow shorter boxes (far-away/seated people)
      )
    ).map(p => {
      const [x, y, w, h] = p.bbox;
      return {
        cx: x + w / 2,
        cy: y + h / 2,
        box: p.bbox,
        score: p.score
      };
    });

    const maxDist = Math.max(120, diagLen * MATCH_RATIO);

    const active = activeRef.current;
    const ghost  = ghostRef.current;

    // ── 2. Expire old ghosts ───────────────────────────────────────────────
    // FIX: Compare real timestamps so ghost TTL is always exactly 3 minutes
    // regardless of how fast or slow the detection loop is running.
    const nowMs = Date.now();
    for (const [id, g] of ghost) {
      if (nowMs - g.ghostedAt > GHOST_TTL_MS) ghost.delete(id);
    }

    // ── 3. Match dets → active tracks (2-pass) ─────────────────────────────
    const pass1 = greedyMatch(active, dets, maxDist);
    const pass2 = greedyMatch(
      new Map(pass1.unmatchedTracks),
      pass1.unmatchedDets,
      maxDist * 2   // wider net for fast movers
    );

    // ── 4. Update matched tracks ───────────────────────────────────────────
    for (const [id, det] of [...pass1.matched, ...pass2.matched]) {
      const prev     = active.get(id);
      const appeared = (prev.appeared || 0) + 1;

      const updatedTrack = { ...prev, cx: det.cx, cy: det.cy, box: det.box, appeared, disappeared: 0 };

      // We need Face-ID if we don't have one, or if we currently only have a Local fallback ID
      // BUG FIX: Don't use !prev.personId because 0 is falsy in JS!
      const needsFaceId = prev.personId === undefined || prev.personId === null || prev.localOnly;
      const now = Date.now();
      const timeSinceLastIdentify = now - (prev.lastIdentifyAttempt || 0);

      // Try Face-ID if stable, and at least 1500ms has passed since last attempt
      if (needsFaceId && appeared >= MIN_FRAMES && timeSinceLastIdentify > 1500) {
        // Safety: if identifying has been stuck for too long, force-release it
        if (prev.identifying && prev.identifyStartTime && (now - prev.identifyStartTime > IDENTIFY_TIMEOUT)) {
          updatedTrack.identifying = false;
        }

        if (!updatedTrack.identifying) {
          // Mark as identifying so we don't trigger multiple requests
          updatedTrack.identifying = true;
          updatedTrack.identifyStartTime = now;
          updatedTrack.lastIdentifyAttempt = now;
          active.set(id, updatedTrack);

          // Asynchronously ask backend (with isMirrored fix)
          identifyPerson(video, det.box, isMirrored).then(result => {
            const currentActive = activeRef.current;
            const currentTrack = currentActive.get(id);

            // ── CASE 1: Backend returned a valid person ID ──
            if (result && result.id !== null && result.id !== undefined && result.status !== 'offline') {
              setBackendOnline(true);
              const backendId = result.id;
              const isRecognized = result.status === 'recognized';

              if (currentTrack) {
                const wasLocalOnly = currentTrack.localOnly;
                
                currentActive.set(id, {
                  ...currentTrack,
                  personId: backendId,
                  counted: true,
                  identifying: false,
                  reidentified: isRecognized,
                  localOnly: false // Upgraded to proper Face-ID!
                });

                // If they were previously counted as Local, but they are actually an old returning person:
                // We overcounted them locally. We must decrement the total count!
                if (wasLocalOnly && isRecognized) {
                  setTotalCount(n => Math.max(0, n - 1));
                  const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                  setLog(prevLogs => [{ id: backendId, t, source: 'face-id', note: '(merged from local)' }, ...prevLogs].slice(0, 50));
                }

                // If this is a BRAND NEW person (not seen before in Face-ID db):
                if (!countedIds.current.has(backendId)) {
                  countedIds.current.add(backendId);
                  
                  if (!wasLocalOnly) {
                    // They weren't counted yet. Increment count!
                    setTotalCount(n => n + 1);
                    setJustCounted(true);
                    clearTimeout(countAnimRef.current);
                    countAnimRef.current = setTimeout(() => setJustCounted(false), 700);
                    const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                    setLog(prevLogs => [{ id: backendId, t, source: 'face-id' }, ...prevLogs].slice(0, 50));
                  } else {
                    // They were ALREADY counted locally. Count stays the same, just log the upgrade.
                    const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                    setLog(prevLogs => [{ id: backendId, t, source: 'face-id', note: '(upgraded to face-id)' }, ...prevLogs].slice(0, 50));
                  }
                }
              } else {
                // They became a ghost while we were identifying
                const ghostRegistry = ghostRef.current;
                const g = ghostRegistry.get(id);
                if (g) {
                  ghostRegistry.set(id, {
                    ...g,
                    personId: backendId,
                    counted: true,
                    reidentified: isRecognized,
                    localOnly: false
                  });
                }
              }

            // ── CASE 2: Backend offline ──
            } else if (result && result.status === 'offline') {
              setBackendOnline(false);
              if (currentTrack) {
                if (!currentTrack.counted) {
                  // FALLBACK: Count them locally if the backend is dead and they haven't been counted
                  const localId = localNextPersonId.current++;
                  currentActive.set(id, {
                    ...currentTrack,
                    personId: localId,
                    counted: true,
                    identifying: false,
                    localOnly: true // Mark as a local fallback
                  });
                  
                  // Add to total count
                  setTotalCount(n => n + 1);
                  setJustCounted(true);
                  clearTimeout(countAnimRef.current);
                  countAnimRef.current = setTimeout(() => setJustCounted(false), 700);
                  
                  const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                  setLog(prevLogs => [{ id: localId, t, source: 'local', note: '(backend offline)' }, ...prevLogs].slice(0, 50));
                } else {
                  // They were already counted locally, just reset identifying flag so we don't spam the offline backend
                  currentActive.set(id, { ...currentTrack, identifying: false });
                }
              }

            // ── CASE 3: Backend said "no_face" or timed out ──
            } else {
              if (currentTrack) {
                const attempts = (currentTrack.noFaceAttempts || 0) + 1;
                if (!currentTrack.counted && attempts >= MAX_NO_FACE_ATTEMPTS) {
                  // Person has been in frame multiple times but face never detected.
                  // Count them locally so they are never missed.
                  const localId = localNextPersonId.current++;
                  currentActive.set(id, {
                    ...currentTrack,
                    personId: localId,
                    counted: true,
                    identifying: false,
                    noFaceAttempts: 0,
                    localOnly: true
                  });
                  setTotalCount(n => n + 1);
                  setJustCounted(true);
                  clearTimeout(countAnimRef.current);
                  countAnimRef.current = setTimeout(() => setJustCounted(false), 700);
                  const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                  setLog(prevLogs => [{ id: localId, t, source: 'local', note: '(counted after retry)' }, ...prevLogs].slice(0, 50));
                } else {
                  // Still has attempts left — retry identify after a short delay
                  currentActive.set(id, { ...currentTrack, identifying: false, noFaceAttempts: attempts });
                }
              }
            }
          }).catch(() => {
            // Network error or timeout — treat like a no_face attempt
            const currentTrack = activeRef.current.get(id);
            if (currentTrack) {
              const attempts = (currentTrack.noFaceAttempts || 0) + 1;
              if (!currentTrack.counted && attempts >= MAX_NO_FACE_ATTEMPTS) {
                const localId = localNextPersonId.current++;
                activeRef.current.set(id, {
                  ...currentTrack,
                  personId: localId,
                  counted: true,
                  identifying: false,
                  noFaceAttempts: 0,
                  localOnly: true
                });
                setTotalCount(n => n + 1);
                setJustCounted(true);
                clearTimeout(countAnimRef.current);
                countAnimRef.current = setTimeout(() => setJustCounted(false), 700);
                const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                setLog(prevLogs => [{ id: localId, t, source: 'local', note: '(timeout fallback)' }, ...prevLogs].slice(0, 50));
              } else {
                activeRef.current.set(id, { ...currentTrack, identifying: false, noFaceAttempts: attempts });
              }
            }
          });
          // FIX (Issue #4): Guard clause instead of 'continue' — the identify
          // call was already fired above; skip the track update below so we
          // don't overwrite the identifying=true state we just set.
          // The previous pattern (fire-then-continue) worked but was confusing
          // because the 'continue' appeared to be inside the .then() callback.
        } else {
          // Already identifying — update position only, leave identifying=true
          active.set(id, updatedTrack);
        }
      } else {
        // Outer else: needsFaceId=false (already counted), OR appeared < MIN_FRAMES,
        // OR within 1500ms cooldown. We MUST write updatedTrack in all these cases
        // so the 'appeared' counter keeps incrementing each frame.
        // Without this line the track is frozen and identify never triggers.
        active.set(id, updatedTrack);
      }
      // Note: tracks where identifying was fired are NOT written here;
      // their state is managed exclusively inside the .then()/.catch() above.
    }

    // ── 5. Unmatched active tracks: age or ghost ───────────────────────────
    const allUnmatched = new Set([
      ...pass1.unmatchedTracks.map(([id]) => id),
      ...pass2.unmatchedTracks.map(([id]) => id),
    ]);
    for (const id of allUnmatched) {
      const t = active.get(id);
      if (!t) continue;
      const disappeared = t.disappeared + 1;
      if (disappeared > MAX_DISAPPEARED) {
        // Move to ghost registry so re-entry is matched back to same ID
        ghost.set(id, { ...t, ghostedAt: Date.now() });
        active.delete(id);
      } else {
        active.set(id, { ...t, disappeared });
      }
    }

    // ── 6. Handle unmatched detections ──────────────────────────────────────
    const finalUnmatched = pass2.unmatchedDets;

    for (const det of finalUnmatched) {

      // ---------------------------------------------------
      // STEP 1: Try matching with ghost memory
      // ---------------------------------------------------

      let matchedGhostId = null;
      let bestGhostDistance = Infinity;

      for (const [id, g] of ghost.entries()) {

        const d = euclidean(g.cx, g.cy, det.cx, det.cy);

        if (d < ghostDist && d < bestGhostDistance) {
          bestGhostDistance = d;
          matchedGhostId = id;
        }
      }

      // ---------------------------------------------------
      // STEP 2: Revive old person
      // ---------------------------------------------------

      if (matchedGhostId !== null) {
        const revivedTrack = ghost.get(matchedGhostId);
        ghost.delete(matchedGhostId);

        active.set(matchedGhostId, {
          ...revivedTrack,
          cx: det.cx,
          cy: det.cy,
          box: det.box,
          appeared: MIN_FRAMES,
          disappeared: 0,
          // If this person was already counted, mark as re-identified so the badge
          // immediately shows 'Already Detected' on re-entry, without waiting for Face-ID.
          reidentified: revivedTrack.counted ? true : revivedTrack.reidentified,
        });

        continue;
      }

      // ---------------------------------------------------
      // STEP 3: Duplicate detection guard
      // ---------------------------------------------------

      let duplicate = false;

      for (const [, t] of active.entries()) {

        const d = euclidean(t.cx, t.cy, det.cx, det.cy);

        const overlap = boxOverlap(t.box, det.box);

        if (
          d < diagLen * DUPE_RATIO ||
          overlap.iou > 0.55 ||
          overlap.iom > 0.75
        ) {
          duplicate = true;
          break;
        }
      }

      if (duplicate) continue;

      // ---------------------------------------------------
      // STEP 4: Create brand new person (with temporary local ID)
      // ---------------------------------------------------

      const newId = 'tmp_' + nextId.current++;

      active.set(newId, {
        cx: det.cx,
        cy: det.cy,
        box: det.box,
        appeared: 1,
        disappeared: 0,
        counted: false,
        identifying: false,
        noFaceAttempts: 0
      });
    }

    // ── 7. UI stats ────────────────────────────────────────────────────────
    setInFrame(
      [...active.values()].filter(
        t => t.disappeared < 10
      ).length
    );
    setGhosts(ghost.size);

    // ── 8. Draw ────────────────────────────────────────────────────────────
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, vW, vH);

    // BUG FIX: Mirror the canvas context when webcam is in selfie mode.
    // The <Webcam mirrored> prop only applies CSS scaleX(-1) to the <video>
    // element — the raw pixel coordinates from COCO-SSD are NOT flipped.
    // We must flip the canvas drawing context to match what the user sees.
    if (isMirrored) {
      ctx.save();
      ctx.translate(vW, 0);
      ctx.scale(-1, 1);
    }

    // Ghost rings (faint purple dots)
    for (const [, g] of ghost) {
      ctx.save();
      ctx.strokeStyle = 'rgba(168,85,247,0.25)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.arc(g.cx, g.cy, 20, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Active tracks
    for (const [, track] of active) {
      const { cx, cy, box, appeared, counted, disappeared, identifying } = track;
      const [bx, by, bw, bh] = box;
      if (disappeared >= MAX_DISAPPEARED) continue;

      const fade  = disappeared === 0 ? 1 : Math.max(0.2, 1 - disappeared / MAX_DISAPPEARED);
      const color = counted ? `rgba(0,230,180,${fade})` : `rgba(255,190,50,${fade})`;

      // Box
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2.5;
      ctx.shadowColor = color;
      ctx.shadowBlur  = 14;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.restore();

      // Scan progress bar (while not yet counted)
      // FIX (Issue #6): Three distinct visual states:
      //   1. Scanning  (appeared < MIN_FRAMES): bar fills left→right
      //   2. Identifying (appeared >= MIN_FRAMES, backend pending): pulsing shimmer
      //   3. Counted: bar hidden
      if (!counted) {
        ctx.save();
        // Background track
        ctx.fillStyle = 'rgba(255,190,50,0.12)';
        ctx.fillRect(bx, by + bh - 5, bw, 5);

        if (!identifying) {
          // Phase 1 — filling progress bar
          const pct = Math.min(appeared / MIN_FRAMES, 1);
          ctx.fillStyle = '#ffbe32';
          ctx.fillRect(bx, by + bh - 5, bw * pct, 5);
        } else {
          // Phase 2 — pulsing shimmer: a bright segment sweeps back and forth
          const t = (Date.now() % 1200) / 1200;          // 0 → 1 in 1.2 s
          const ping = Math.abs(Math.sin(t * Math.PI)); // 0→1→0 (ping-pong)
          const seg  = bw * 0.4;                         // shimmer width = 40% of box
          const startX = bx + (bw - seg) * ping;
          ctx.fillStyle = 'rgba(100,200,255,0.7)';
          ctx.fillRect(startX, by + bh - 5, seg, 5);
        }
        ctx.restore();
      }

      // Centroid dot
      ctx.save();
      ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
      ctx.restore();

      // ID badge — BUG FIX: When mirrored, text would render backwards.
      // We temporarily un-flip for text rendering by saving/restoring context.
      if (isMirrored) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-vW, 0);
        const flippedBx = vW - bx - bw;
        let label = `⏳ Scanning...`;
        if (counted) {
          const pId = track.personId !== undefined ? track.personId : '?';
          const displayId = track.localOnly ? `L-${pId - 10000}` : `${pId}`;
          if (track.reidentified) {
            label = `↩ Already Detected (ID ${displayId})`;
          } else if (track.localOnly) {
            label = disappeared > 0 ? `↩ ID ${displayId} ⚡` : `✓ ID ${displayId} ⚡`;
          } else {
            label = disappeared > 0 ? `↩ Already Detected (ID ${displayId})` : `✓ ID ${displayId}`;
          }
        } else if (identifying) {
          label = `🔍 Identifying...`;
        }
        const badgeC = counted
          ? `rgba(0,230,180,${Math.max(0.8, fade)})`
          : `rgba(255,190,50,${Math.max(0.8, fade)})`;
        drawBadge(ctx, flippedBx, by - 28, label, badgeC);
        ctx.restore();
      } else {
        let label = `⏳ Scanning...`;
        if (counted) {
          const pId = track.personId !== undefined ? track.personId : '?';
          const displayId = track.localOnly ? `L-${pId - 10000}` : `${pId}`;
          if (track.reidentified) {
            label = `↩ Already Detected (ID ${displayId})`;
          } else if (track.localOnly) {
            label = disappeared > 0 ? `↩ ID ${displayId} ⚡` : `✓ ID ${displayId} ⚡`;
          } else {
            label = disappeared > 0 ? `↩ Already Detected (ID ${displayId})` : `✓ ID ${displayId}`;
          }
        } else if (identifying) {
          label = `🔍 Identifying...`;
        }
        const badgeC = counted
          ? `rgba(0,230,180,${Math.max(0.8, fade)})`
          : `rgba(255,190,50,${Math.max(0.8, fade)})`;
        drawBadge(ctx, bx, by - 28, label, badgeC);
      }
    }

    // BUG FIX: Restore context if we flipped it for mirrored mode
    if (isMirrored) {
      ctx.restore();
    }
    } finally {
      isDetectingRef.current = false;
    }
  }, [facing]);

  // ── RAF loop ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!modelReady) return;
    const loop = ts => {
      rafRef.current = requestAnimationFrame(loop);
      if (ts - lastTickRef.current < DETECT_MS) return;
      lastTickRef.current = ts;
      detect();
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [modelReady, detect]);

  // ── Reset ───────────────────────────────────────────────────────────────────
  const handleReset = () => {
    // Clear in-component refs AND the module-level singletons so they stay in
    // sync — a remount after reset must not see stale singleton data.
    activeRef.current.clear();    _active.clear();
    ghostRef.current.clear();     _ghost.clear();
    countedIds.current.clear();   _countedIds.clear();
    nextId.current = 0;           _nextId = 0;
    // FIX: Reset to 10000, not 0 — preserves the offset guard that prevents
    // local fallback IDs from colliding with backend Face-IDs (0, 1, 2…).
    localNextPersonId.current = 10000;  _localNextId = 10000;
    // FIX: Clear the animation timer ref on reset to avoid stale callbacks.
    clearTimeout(countAnimRef.current);
    countAnimRef.current = null;
    setTotalCount(0);
    setInFrame(0);
    setGhosts(0);
    setLog([]);
    const c = canvasRef.current;
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);

    // Also clear the Python DeepFace face database
    fetch('/api/reset', { method: 'POST' })
      .then(() => console.log('Backend face database cleared!'))
      .catch(err => console.warn('Could not reset backend (offline?):', err));
  };

  const toggleCamera = () => {
    cancelAnimationFrame(rafRef.current);
    handleReset();
    setFacing(f => f === 'user' ? 'environment' : 'user');
  };

  // ── Render ──────────────────────────────────────────────────────────────────
  // Compute ring stroke offset (440 = full circumference for r=70)
  // Fill from 0 up to max 50 people; cap at 440 (full circle)
  const ringOffset = Math.max(0, 440 - Math.min(totalCount / 50, 1) * 440);

  return (
    <div className="pc-root">
      <header className="pc-header">
        <div className="pc-header-left">
          <span className="pc-logo">👁‍🗨</span>
          <div>
            <h1 className="pc-title">VisionCount AI</h1>
            <p className="pc-subtitle">Real-time Human Detection &amp; Counting</p>
          </div>
        </div>
        <div className="pc-header-right">
          {/* Session timer */}
          <div className="pc-session-badge">
            <span style={{fontSize:'0.7rem', color:'var(--text-muted)'}}>⏱</span>
            <span className="pc-session-time">{fmtTime(sessionTime)}</span>
          </div>
          <div className="pc-stat-mini">
            <span className="pc-stat-mini-val">{inFrame}</span>
            <span className="pc-stat-mini-label">In Frame</span>
          </div>
          <div className="pc-stat-mini ghost-stat">
            <span className="pc-stat-mini-val">{ghosts}</span>
            <span className="pc-stat-mini-label">Memory</span>
          </div>
          <div className={`pc-status-pill ${backendOnline ? '' : 'pc-status-offline'}`}>
            <span className={`pc-dot ${backendOnline ? 'active' : 'offline-dot'}`} />
            {backendOnline ? 'Face-ID Active' : 'Local Mode ⚡'}
          </div>
          <div className="pc-status-pill">
            <span className={`pc-dot ${modelReady ? 'active' : 'loading'}`} />
            {modelReady ? 'AI Ready' : 'Loading…'}
          </div>
        </div>
      </header>

      <main className="pc-main">
        <div className="pc-camera-wrap">
          {camError ? (
            <div className="pc-cam-error">
              <span>📷</span>
              <p>Camera access denied or unavailable.</p>
              <p className="pc-cam-error-sub">Allow camera permission and refresh the page.</p>
            </div>
          ) : (
            <>
              <Webcam
                ref={webcamRef}
                className="pc-webcam"
                audio={false}
                mirrored={facing === 'user'}
                videoConstraints={{ facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 } }}
                onUserMediaError={() => setCamError(true)}
              />
              <canvas ref={canvasRef} className="pc-canvas" />
              {/* Scan-line texture */}
              <div className="pc-camera-scanline" />
              {/* Moving scan bar */}
              {modelReady && <div className="pc-camera-scanbar" />}
              {/* Extra corner brackets */}
              <div className="pc-corner-tr" />
              <div className="pc-corner-bl" />
              {/* REC badge */}
              {modelReady && (
                <div className="pc-rec-badge">
                  <span className="pc-rec-dot" />
                  LIVE
                </div>
              )}
              {!modelReady && (
                <div className="pc-overlay-loading">
                  <div className="pc-spinner" />
                  <p className="pc-loading-title">Initializing AI Engine</p>
                  <p className="pc-loading-sub">Loading COCO-SSD model… ~10 s on first load</p>
                </div>
              )}
            </>
          )}
        </div>

        <aside className="pc-panel">
          {/* Count card with SVG ring */}
          <div className={`pc-card pc-count-card ${justCounted ? 'counted-anim' : ''}`}>
            <p className="pc-card-label">Total Unique People</p>
            <div className="pc-count-ring">
              <svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#00f2fe" />
                    <stop offset="100%" stopColor="#4facfe" />
                  </linearGradient>
                </defs>
                <circle className="pc-count-ring-track" cx="80" cy="80" r="70" />
                <circle
                  className="pc-count-ring-fill"
                  cx="80" cy="80" r="70"
                  style={{ strokeDashoffset: ringOffset }}
                />
              </svg>
              <div className="pc-count-display">{totalCount}</div>
            </div>
            <p className="pc-card-hint">Each person counted exactly once — even on re-entry</p>
          </div>

          {/* Live status */}
          <div className="pc-card pc-live-status">
            <p className="pc-card-label">Live Status</p>
            <div className="pc-live-row">
              <div className="pc-live-item teal-item">
                <span className="pc-live-num">{inFrame}</span>
                <span className="pc-live-sub">In Frame</span>
              </div>
              <div className="pc-live-item purple-item">
                <span className="pc-live-num">{ghosts}</span>
                <span className="pc-live-sub">Ghost Memory</span>
              </div>
            </div>
          </div>

          {/* Activity log */}
          <div className="pc-card pc-log-card">
            <p className="pc-card-label">Activity Log</p>
            {log.length === 0 ? (
              <p className="pc-empty-log">No detections yet…</p>
            ) : (
              <ul className="pc-log-list">
                {log.map((entry, i) => (
                  <li key={`${entry.id}-${i}`} className={`pc-log-item ${i === 0 ? 'pc-log-latest' : ''}`}>
                    <span className="pc-log-check">{entry.source === 'local' ? '⚡' : '✓'}</span>
                    <span className="pc-log-text">
                      Person <strong>ID {entry.id}</strong> counted
                      {entry.source === 'local' && <span className="pc-log-local"> (local)</span>}
                      {entry.note && <span style={{fontSize:'0.65rem',color:'var(--text-dim)',marginLeft:4}}>{entry.note}</span>}
                    </span>
                    <span className="pc-log-time">{entry.t}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Legend */}
          <div className="pc-card pc-legend">
            <p className="pc-card-label">Canvas Legend</p>
            <ul className="pc-legend-list">
              <li><span className="legend-swatch teal" />Counted person</li>
              <li><span className="legend-swatch amber" />Scanning — confirming</li>
              <li><span className="legend-swatch purple" />Ghost memory position</li>
            </ul>
          </div>

          {/* Controls */}
          <div className="pc-card pc-controls">
            <p className="pc-card-label">Controls</p>
            <button className="pc-btn pc-btn-danger"    onClick={handleReset}>🔄 Reset Count &amp; Memory</button>
            <button className="pc-btn pc-btn-secondary" onClick={toggleCamera}>🔁 Switch Camera</button>
          </div>

          {/* How it works */}
          <div className="pc-card pc-info">
            <p className="pc-card-label">How It Works</p>
            <ol className="pc-info-list">
              <li><strong>Person-only:</strong> COCO-SSD detects bodies with ≥35% confidence.</li>
              <li><strong>NMS:</strong> Overlapping boxes collapsed to one per body.</li>
              <li><strong>2-pass tracking:</strong> Tight then wide centroid matching per frame.</li>
              <li><strong>Ghost memory:</strong> People remembered for ~3 min after leaving.</li>
              <li><strong>Face-ID:</strong> DeepFace backend provides identity across sessions.</li>
            </ol>
          </div>
        </aside>
      </main>
    </div>
  );
}
