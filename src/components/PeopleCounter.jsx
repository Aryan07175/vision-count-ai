import { useEffect, useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Capacitor } from '@capacitor/core';
import './PeopleCounter.css';

// ─── CONSTANTS ─────────────────────────────────────────────────────────────────
const CONFIDENCE      = 0.40;
const NMS_IOU         = 0.50;
const NMS_IOM         = 0.85;
const MIN_FRAMES      = 5;
const MAX_DISAPPEARED = 10;
const GHOST_TTL       = 3;
const MATCH_RATIO     = 0.08;
const GHOST_RATIO     = 0.05;
const DUPE_RATIO      = 0.05;
const DETECT_MS       = 150;
const IDENTIFY_TIMEOUT = 10000; // 10 s max wait for DeepFace

// Backend URL for native Capacitor builds — configure via VITE_BACKEND_URL in .env
const NATIVE_BACKEND = (import.meta.env.VITE_BACKEND_URL ?? 'http://192.168.1.10:8000');

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
  const matched = [];

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
  ctx.fillStyle = 'rgba(0,0,0,0.85)';
  ctx.fillText(label, x + 8, y + 15);
  ctx.restore();
}

// Format seconds → HH:MM:SS
function formatDuration(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  return [h, m, s].map(n => String(n).padStart(2, '0')).join(':');
}

// ─── API HELPER ───────────────────────────────────────────────────────────────
async function identifyPerson(video, box) {
  const [x, y, w, h] = box;
  const vW = video.videoWidth;
  const vH = video.videoHeight;

  const margin = Math.min(w, h) * 0.1;
  const cropX = Math.max(0, x - margin);
  const cropY = Math.max(0, y - margin);
  const cropW = Math.min(vW - cropX, w + margin * 2);
  const cropH = Math.min(vH - cropY, h + margin * 2);

  const canvas = document.createElement('canvas');
  canvas.width  = cropW;
  canvas.height = cropH;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
  const base64 = canvas.toDataURL('image/jpeg', 0.8);

  try {
    const isNative = Capacitor.isNativePlatform();
    const endpoint = isNative
      ? `${NATIVE_BACKEND}/api/identify`
      : '/api/identify';

    const controller = new AbortController();
    const timeoutId  = setTimeout(() => controller.abort(), IDENTIFY_TIMEOUT);

    const res = await fetch(endpoint, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image_base64: base64 }),
      signal:  controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) return { error: 'backend_error' };
    return await res.json();
  } catch (err) {
    console.warn('DeepFace API error:', err);
    return { error: 'network' };
  }
}

// ─── COMPONENT ────────────────────────────────────────────────────────────────
export default function PeopleCounter() {
  const webcamRef      = useRef(null);
  const canvasRef      = useRef(null);
  const rafRef         = useRef(null);
  const lastTickRef    = useRef(0);
  const isDetectingRef = useRef(false);
  const modelRef       = useRef(null);

  // FIX: use ref instead of window.countAnimTimeout to avoid global leak
  const countAnimTimeoutRef = useRef(null);

  // Active tracks:  Map<id, { cx, cy, box, appeared, disappeared, counted }>
  const activeRef   = useRef(new Map());
  // Ghost registry: Map<id, { cx, cy, frame }>
  const ghostRef    = useRef(new Map());
  // Permanent set of every backend ID ever counted
  const countedIds  = useRef(new Set());

  const nextId     = useRef(0);
  const frameCount = useRef(0);

  const [totalCount,  setTotalCount]  = useState(0);
  const [justCounted, setJustCounted] = useState(false);
  const [modelReady,  setModelReady]  = useState(false);
  const [camError,    setCamError]    = useState(false);
  const [facing,      setFacing]      = useState('user');
  const [inFrame,     setInFrame]     = useState(0);
  const [ghosts,      setGhosts]      = useState(0);
  const [log,         setLog]         = useState([]);
  const [apiError,    setApiError]    = useState(false);
  const [globalError, setGlobalError] = useState(null);

  // Session timer
  const sessionStartRef = useRef(null);
  const [sessionSecs, setSessionSecs] = useState(0);

  // ── Global error handler ───────────────────────────────────────────────────
  // FIX: capture the handler ref so it can be removed in cleanup
  useEffect(() => {
    const rejectionHandler = (e) =>
      setGlobalError(e.reason?.message || 'Unhandled promise rejection');

    window.onerror = (msg, _url, lineNo, colNo) => {
      setGlobalError(`${msg} (${lineNo}:${colNo})`);
      return false;
    };
    window.addEventListener('unhandledrejection', rejectionHandler);

    return () => {
      window.onerror = null;
      window.removeEventListener('unhandledrejection', rejectionHandler);
    };
  }, []);

  // ── Session timer tick ─────────────────────────────────────────────────────
  useEffect(() => {
    if (sessionStartRef.current === null) {
      sessionStartRef.current = Date.now();
    }
    const id = setInterval(() => {
      if (sessionStartRef.current !== null) {
        setSessionSecs(Math.floor((Date.now() - sessionStartRef.current) / 1000));
      }
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // ── Load TF + COCO-SSD model ───────────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    (async () => {
      await tf.ready();
      if (!alive) return;
      modelRef.current = await cocoSsd.load({ base: 'mobilenet_v2' });
      if (alive) setModelReady(true);
    })();
    return () => {
      alive = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // ── Detection loop ──────────────────────────────────────────────────────────
  const detect = useCallback(async () => {
    if (isDetectingRef.current) return;
    isDetectingRef.current = true;
    try {
      const model  = modelRef.current;
      const webcam = webcamRef.current;
      const canvas = canvasRef.current;
      if (!model || !webcam || !canvas) return;
      const video = webcam.video;
      if (!video || video.readyState !== 4) return;
      const vW = video.videoWidth, vH = video.videoHeight;
      if (!vW || !vH) return;

      canvas.width = vW; canvas.height = vH;
      const diagLen   = Math.hypot(vW, vH);
      const ghostDist = diagLen * GHOST_RATIO;
      const frame     = ++frameCount.current;

      // ── 1. Detect persons only ───────────────────────────────────────────
      const raw  = await model.detect(video);
      const dets = applyNMS(
        raw.filter(
          p =>
            typeof p.class === 'string' &&
            p.class.trim().toLowerCase() === 'person' &&
            p.score >= CONFIDENCE &&
            p.bbox[2] > 20 &&
            p.bbox[3] > 40,
        ),
      ).map(p => {
        const [x, y, w, h] = p.bbox;
        return { cx: x + w / 2, cy: y + h / 2, box: p.bbox };
      });

      const maxDist = diagLen * MATCH_RATIO;
      const active  = activeRef.current;
      const ghost   = ghostRef.current;

      // ── 2. Expire old ghosts ─────────────────────────────────────────────
      for (const [id, g] of ghost) {
        if (frame - g.frame > GHOST_TTL) ghost.delete(id);
      }

      // ── 3. Match dets → active tracks ────────────────────────────────────
      const pass1 = greedyMatch(active, dets, maxDist);

      // ── 4. Update matched tracks ──────────────────────────────────────────
      for (const [id, det] of pass1.matched) {
        const prev     = active.get(id);
        const appeared = (prev.appeared || 0) + 1;
        const alreadyCounted = !!prev.counted;

        const updatedTrack = {
          ...prev, cx: det.cx, cy: det.cy, box: det.box, appeared, disappeared: 0,
        };

        if (!alreadyCounted && appeared >= MIN_FRAMES) {
          // Safety: release stuck identification
          if (
            prev.identifying &&
            prev.identifyStartTime &&
            (Date.now() - prev.identifyStartTime > IDENTIFY_TIMEOUT)
          ) {
            updatedTrack.identifying = false;
          }

          if (!updatedTrack.identifying) {
            updatedTrack.identifying       = true;
            updatedTrack.identifyStartTime = Date.now();
            active.set(id, updatedTrack);

            identifyPerson(video, det.box).then(result => {
              const currentActive  = activeRef.current;
              const ghostRegistry  = ghostRef.current;

              let trackToUpdate = currentActive.get(id);
              let isGhost = false;

              if (!trackToUpdate) {
                trackToUpdate = ghostRegistry.get(id);
                isGhost = true;
              }
              if (!trackToUpdate) return; // Completely expired

              // ── CASE 1: Backend returned a valid person ID ──────────────
              if (result && result.id !== null && result.id !== undefined) {
                const backendId = result.id;
                setApiError(false);

                trackToUpdate = {
                  ...trackToUpdate,
                  personId:     backendId,
                  counted:      true,
                  identifying:  false,
                  reidentified: result.status === 'recognized',
                };

                if (isGhost) ghostRegistry.set(id, trackToUpdate);
                else currentActive.set(id, trackToUpdate);

                if (!countedIds.current.has(backendId)) {
                  countedIds.current.add(backendId);
                  setTotalCount(n => n + 1);
                  setJustCounted(true);
                  // FIX: use ref instead of window global
                  clearTimeout(countAnimTimeoutRef.current);
                  countAnimTimeoutRef.current = setTimeout(
                    () => setJustCounted(false), 700,
                  );
                  const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                  setLog(prev => [{ id: backendId, t }, ...prev].slice(0, 15));
                }

              // ── CASE 2: Network error (offline) ─────────────────────────
              } else if (result && result.error === 'network') {
                setApiError(true);
                if (!trackToUpdate.counted) {
                  const anonId = 'Anon-' + String(id).replace('tmp_', '');

                  trackToUpdate = {
                    ...trackToUpdate,
                    personId:    anonId,
                    counted:     true,
                    identifying: false,
                    reidentified: false,
                  };

                  if (isGhost) ghostRegistry.set(id, trackToUpdate);
                  else currentActive.set(id, trackToUpdate);

                  setTotalCount(n => n + 1);
                  setJustCounted(true);
                  clearTimeout(countAnimTimeoutRef.current);
                  countAnimTimeoutRef.current = setTimeout(
                    () => setJustCounted(false), 700,
                  );
                  const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                  setLog(prev => [{ id: anonId, t }, ...prev].slice(0, 15));
                }

              // ── CASE 3: No face / null — fallback after ~4.5 s ──────────
              } else {
                if (!trackToUpdate.counted && trackToUpdate.appeared > 30) {
                  const anonId = 'Anon-' + String(id).replace('tmp_', '');

                  trackToUpdate = {
                    ...trackToUpdate,
                    personId:    anonId,
                    counted:     true,
                    identifying: false,
                    reidentified: false,
                  };

                  if (isGhost) ghostRegistry.set(id, trackToUpdate);
                  else currentActive.set(id, trackToUpdate);

                  setTotalCount(n => n + 1);
                  setJustCounted(true);
                  clearTimeout(countAnimTimeoutRef.current);
                  countAnimTimeoutRef.current = setTimeout(
                    () => setJustCounted(false), 700,
                  );
                  const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                  setLog(prev => [{ id: anonId, t }, ...prev].slice(0, 15));
                } else {
                  // Keep trying next frame
                  trackToUpdate.identifying = false;
                  if (isGhost) ghostRegistry.set(id, trackToUpdate);
                  else currentActive.set(id, trackToUpdate);
                }
              }
            }).catch(err => {
              console.warn('Unexpected identify error:', err);
              const t = activeRef.current.get(id);
              if (t) activeRef.current.set(id, { ...t, identifying: false });
            });

            continue; // Skip the active.set below — already done above
          }
        }

        active.set(id, updatedTrack);
      }

      // ── 5. Unmatched active tracks: age or ghost ──────────────────────────
      for (const [id] of pass1.unmatchedTracks) {
        const t = active.get(id);
        if (!t) continue;
        const disappeared = t.disappeared + 1;
        if (disappeared > MAX_DISAPPEARED) {
          ghost.set(id, { ...t, frame });
          active.delete(id);
        } else {
          active.set(id, { ...t, disappeared });
        }
      }

      // ── 6. Handle unmatched detections ────────────────────────────────────
      for (const det of pass1.unmatchedDets) {
        // Try ghost memory first
        let matchedGhostId   = null;
        let bestGhostDist    = Infinity;

        for (const [id, g] of ghost.entries()) {
          const d = euclidean(g.cx, g.cy, det.cx, det.cy);
          if (d < ghostDist && d < bestGhostDist) {
            bestGhostDist  = d;
            matchedGhostId = id;
          }
        }

        if (matchedGhostId !== null) {
          const revivedTrack = ghost.get(matchedGhostId);
          ghost.delete(matchedGhostId);
          active.set(matchedGhostId, {
            ...revivedTrack,
            cx: det.cx, cy: det.cy, box: det.box,
            appeared: MIN_FRAMES, disappeared: 0,
          });
          continue;
        }

        // Duplicate guard
        let duplicate = false;
        for (const [, t] of active.entries()) {
          const d = euclidean(t.cx, t.cy, det.cx, det.cy);
          const { iou, iom } = boxOverlap(t.box, det.box);
          if (d < diagLen * DUPE_RATIO || iou > 0.55 || iom > 0.75) {
            duplicate = true;
            break;
          }
        }
        if (duplicate) continue;

        // New person
        const newId = 'tmp_' + nextId.current++;
        active.set(newId, {
          cx: det.cx, cy: det.cy, box: det.box,
          appeared: 1, disappeared: 0,
          counted: false, identifying: false,
        });
      }

      // ── 7. Update UI stats ────────────────────────────────────────────────
      setInFrame([...active.values()].filter(t => t.disappeared < 10).length);
      setGhosts(ghost.size);

      // ── 8. Draw overlay ───────────────────────────────────────────────────
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, vW, vH);
      const isMirrored = facing === 'user';

      // Ghost rings
      for (const [, g] of ghost) {
        ctx.save();
        ctx.strokeStyle = 'rgba(168,85,247,0.3)';
        ctx.lineWidth   = 1.5;
        ctx.setLineDash([3, 5]);
        ctx.shadowColor = 'rgba(168,85,247,0.4)';
        ctx.shadowBlur  = 6;
        ctx.beginPath();
        const drawCx = isMirrored ? vW - g.cx : g.cx;
        ctx.arc(drawCx, g.cy, 20, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // Active tracks
      for (const [, track] of active) {
        const { cx, cy, box, appeared, counted, disappeared, identifying } = track;
        const [bx, by, bw, bh] = box;
        if (disappeared >= MAX_DISAPPEARED) continue;

        const fade  = disappeared === 0 ? 1 : Math.max(0.2, 1 - disappeared / MAX_DISAPPEARED);
        const color = counted
          ? `rgba(0,230,180,${fade})`
          : `rgba(255,190,50,${fade})`;

        const drawBx = isMirrored ? vW - bx - bw : bx;
        const drawCx = isMirrored ? vW - cx       : cx;

        // Bounding box with corner accents
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.shadowColor = color;
        ctx.shadowBlur  = 12;

        // Draw corner brackets instead of full box for a cleaner look
        const cLen = Math.min(bw, bh) * 0.18;
        const corners = [
          [drawBx,        by],
          [drawBx + bw,   by],
          [drawBx,        by + bh],
          [drawBx + bw,   by + bh],
        ];
        const dirs = [
          [1, 1], [-1, 1], [1, -1], [-1, -1],
        ];
        for (let i = 0; i < corners.length; i++) {
          const [cx2, cy2] = corners[i];
          const [dx, dy]   = dirs[i];
          ctx.beginPath();
          ctx.moveTo(cx2 + dx * cLen, cy2);
          ctx.lineTo(cx2, cy2);
          ctx.lineTo(cx2, cy2 + dy * cLen);
          ctx.stroke();
        }

        // Thin full-box at low opacity
        ctx.globalAlpha = 0.25;
        ctx.strokeRect(drawBx, by, bw, bh);
        ctx.globalAlpha = 1;
        ctx.restore();

        // Scan progress bar
        if (!counted) {
          const pct = Math.min(appeared / MIN_FRAMES, 1);
          ctx.save();
          ctx.fillStyle = 'rgba(255,190,50,0.1)';
          ctx.fillRect(drawBx, by + bh - 4, bw, 4);
          ctx.fillStyle = '#ffbe32';
          ctx.shadowColor = '#ffbe32';
          ctx.shadowBlur  = 8;
          ctx.fillRect(drawBx, by + bh - 4, bw * pct, 4);
          ctx.restore();
        }

        // Centroid dot
        ctx.save();
        ctx.fillStyle  = color;
        ctx.shadowColor = color;
        ctx.shadowBlur  = 12;
        ctx.beginPath();
        ctx.arc(drawCx, cy, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // ID badge
        let label = '⏳ Scanning…';
        if (counted) {
          const pId = track.personId !== undefined ? track.personId : '?';
          label = track.reidentified
            ? `↩ RETURNING (ID ${pId})`
            : (disappeared > 0 ? `↩ ID ${pId}` : `✓ ID ${pId}`);
        } else if (identifying) {
          label = '🔍 Identifying…';
        }

        const badgeC = counted
          ? `rgba(0,230,180,${Math.max(0.85, fade)})`
          : `rgba(255,190,50,${Math.max(0.85, fade)})`;
        drawBadge(ctx, drawBx, by - 28, label, badgeC);
      }
    } finally {
      isDetectingRef.current = false;
    }
  }, [facing]);

  // ── RAF loop ─────────────────────────────────────────────────────────────────
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

  // Cleanup timeout on unmount
  useEffect(() => () => clearTimeout(countAnimTimeoutRef.current), []);

  // ── Reset ─────────────────────────────────────────────────────────────────────
  const handleReset = () => {
    activeRef.current.clear();
    ghostRef.current.clear();
    countedIds.current.clear();
    nextId.current    = 0;
    frameCount.current = 0;
    sessionStartRef.current = Date.now();
    setTotalCount(0);
    setInFrame(0);
    setGhosts(0);
    setSessionSecs(0);
    setLog([]);
    setApiError(false);
    setGlobalError(null);

    const c = canvasRef.current;
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);

    // Clear the Python DeepFace face database
    fetch('/api/reset', { method: 'POST' })
      .then(() => console.log('[Reset] Backend face database cleared.'))
      .catch(err => console.warn('[Reset] Could not reach backend:', err));
  };

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div className="pc-root">

      {/* Global error toast — FIX: was set but never rendered */}
      {globalError && (
        <div className="pc-global-error" role="alert">
          <span className="pc-global-error-icon">⚠</span>
          <span>{globalError}</span>
          <button className="pc-global-error-close" onClick={() => setGlobalError(null)}>✕</button>
        </div>
      )}

      <header className="pc-header">
        <div className="pc-header-left">
          <span className="pc-logo">👁‍🗨</span>
          <div>
            <h1 className="pc-title">VisionCount AI</h1>
            <p className="pc-subtitle">Real-time Human Detection &amp; Counting</p>
          </div>
        </div>
        <div className="pc-header-right">
          <div className="pc-stat-mini" title="People currently visible in camera">
            <span className="pc-stat-mini-val">{inFrame}</span>
            <span className="pc-stat-mini-label">In Frame</span>
          </div>
          <div className="pc-stat-mini ghost-stat" title="People tracked outside frame">
            <span className="pc-stat-mini-val">{ghosts}</span>
            <span className="pc-stat-mini-label">Memory</span>
          </div>
          <div className="pc-session-badge" title="Session duration">
            <span className="pc-session-icon">⏱</span>
            <span className="pc-session-time">{formatDuration(sessionSecs)}</span>
          </div>
          <div className="pc-status-pill" title={modelReady ? 'AI model loaded and running' : 'Loading AI model…'}>
            <span className={`pc-dot ${modelReady ? 'active' : 'loading'}`} />
            {modelReady ? 'Model Ready' : 'Loading…'}
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
                videoConstraints={{
                  facingMode: facing,
                  width:  { ideal: 640 },
                  height: { ideal: 480 },
                }}
                onUserMediaError={() => setCamError(true)}
              />
              <button
                className="pc-btn-flip"
                onClick={() => setFacing(f => f === 'user' ? 'environment' : 'user')}
                title="Switch Camera"
              >
                🔄
              </button>
              <canvas ref={canvasRef} className="pc-canvas" />
              {!modelReady && (
                <div className="pc-overlay-loading">
                  <div className="pc-spinner" />
                  <p className="pc-loading-title">Initializing TensorFlow.js</p>
                  <p className="pc-loading-sub">Loading COCO-SSD model… ~10 s on first load</p>
                </div>
              )}
            </>
          )}

          {apiError && (
            <div className="pc-api-error-banner">
              <span>📡</span> Backend unreachable — showing anonymous IDs
            </div>
          )}
        </div>

        <aside className="pc-panel">
          {/* Count card */}
          <div className={`pc-card pc-count-card ${justCounted ? 'counted-anim' : ''}`}>
            <p className="pc-card-label">Total Unique People</p>
            <div className="pc-count-ring">
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
                    <span className="pc-log-check">✓</span>
                    <span className="pc-log-text">Person <strong>ID {entry.id}</strong> counted</span>
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
              <li><span className="legend-swatch teal" /><code>✓ ID</code> — confirmed &amp; counted</li>
              <li><span className="legend-swatch amber" />Scanning — waiting for face ID</li>
              <li><span className="legend-swatch purple" />Ghost — remembered, out of frame</li>
            </ul>
          </div>

          {/* Controls */}
          <div className="pc-card pc-controls">
            <p className="pc-card-label">Controls</p>
            <button className="pc-btn pc-btn-danger" onClick={handleReset}>🔄 Reset Session</button>
          </div>

          {/* How it works */}
          <div className="pc-card pc-info">
            <p className="pc-card-label">How It Works</p>
            <ol className="pc-info-list">
              <li><strong>Person-only:</strong> COCO-SSD detects "person" class ≥ 40% confidence — all other objects ignored.</li>
              <li><strong>NMS:</strong> Overlapping / nested boxes collapsed to one per body.</li>
              <li><strong>Centroid tracking:</strong> Greedy nearest-neighbour match per frame.</li>
              <li><strong>Ghost memory:</strong> People who exit are remembered; re-matched on return — never re-counted.</li>
              <li><strong>Face identity:</strong> DeepFace Facenet512 assigns a permanent unique ID per face.</li>
            </ol>
          </div>
        </aside>
      </main>
    </div>
  );
}
