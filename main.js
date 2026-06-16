/* ============================================================
   AIR DRAW — Main Application
   Gesture-based interactive doodler powered by MediaPipe
   ============================================================ */

// ── State ──────────────────────────────────────────────────────
const state = {
  handLandmarker: null,
  webcamStream: null,
  isReady: false,
  // Drawing
  strokes: [],           // Array of { points: [{x,y}], color, thickness, glow }
  currentStroke: null,
  activeColor: '#00f0ff',
  thickness: 6,
  glowIntensity: 60,
  // Gesture
  currentGesture: 'idle',
  previousGesture: 'idle',
  gestureStableFrames: 0,
  gestureStartTime: 0,
  isModalOpen: true,
  // Grab & Move
  isGrabbing: false,
  grabStartPos: null,
  grabOffset: { x: 0, y: 0 },
  totalOffset: { x: 0, y: 0 },
  nearestStrokeIdx: -1,
  // Erase
  eraserRadius: 28,
  // Camera
  showCamera: true,
  cameraOpacity: 0.35,
  // Particles
  particles: [],
  // Smoothing
  smoothPos: { x: 0, y: 0 },
  smoothFactor: 0.35,
  // Canvas dimensions
  width: 0,
  height: 0,
  // Audio
  audioCtx: null,
};

// ── DOM Elements ───────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const loadingScreen = $('loading-screen');
const appEl = $('app');
const webcamEl = $('webcam');
const cameraCanvas = $('camera-canvas');
const drawingCanvas = $('drawing-canvas');
const uiCanvas = $('ui-canvas');
const cameraCtx = cameraCanvas.getContext('2d');
const drawingCtx = drawingCanvas.getContext('2d');
const uiCtx = uiCanvas.getContext('2d');
const gestureHud = $('gesture-hud');
const gestureIcon = $('gesture-icon');
const gestureLabel = $('gesture-label');
const thicknessSlider = $('thickness-slider');
const thicknessValue = $('thickness-value');
const glowSlider = $('glow-slider');
const glowValue = $('glow-value');

const cameraModeText = $('camera-mode-text');
const cameraModeIndicator = $('camera-mode-indicator');
const onboardingModal = $('onboarding-modal');
const btnStart = $('btn-start');

// ── Audio (subtle sound effects) ───────────────────────────────
function getAudioCtx() {
  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return state.audioCtx;
}

function playTone(freq, duration, type = 'sine', volume = 0.06) {
  try {
    const ctx = getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(volume, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) { /* audio not available */ }
}

function playDrawStart() { playTone(880, 0.08, 'sine', 0.04); }
function playDrawEnd() { playTone(440, 0.1, 'sine', 0.03); }
function playEraseSound() { playTone(200, 0.06, 'triangle', 0.03); }
function playGrabSound() { playTone(660, 0.1, 'sine', 0.05); }
function playDropSound() { playTone(330, 0.15, 'sine', 0.04); }
function playTrashSound() { 
  playTone(400, 0.1, 'sawtooth', 0.04); 
  setTimeout(() => playTone(200, 0.2, 'sawtooth', 0.05), 100);
}
function playModeSwitch() { playTone(1200, 0.05, 'sine', 0.03); }

// ── Canvas Setup ───────────────────────────────────────────────
function resizeCanvases() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  state.width = w;
  state.height = h;
  [cameraCanvas, drawingCanvas, uiCanvas].forEach(c => {
    c.width = w;
    c.height = h;
  });
}

window.addEventListener('resize', () => {
  resizeCanvases();
  redrawStrokes();
});

// ── MediaPipe Loading ──────────────────────────────────────────
async function initMediaPipe() {
  // Dynamic import from CDN
  const { FilesetResolver, HandLandmarker } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs'
  );

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  );

  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });

  return true;
}

// ── Webcam Setup ───────────────────────────────────────────────
async function initWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
  });
  webcamEl.srcObject = stream;
  state.webcamStream = stream;

  return new Promise((resolve) => {
    webcamEl.onloadedmetadata = () => {
      webcamEl.play();
      resolve();
    };
  });
}

// ── Gesture Detection ──────────────────────────────────────────
function detectGesture(landmarks) {
  if (!landmarks || landmarks.length === 0) return 'none';

  const lm = landmarks;

  // Finger tip and pip/mcp landmarks
  const thumbTip = lm[4];
  const thumbIP = lm[3];
  const indexTip = lm[8];
  const indexPIP = lm[6];
  const indexMCP = lm[5];
  const middleTip = lm[12];
  const middlePIP = lm[10];
  const ringTip = lm[16];
  const ringPIP = lm[14];
  const pinkyTip = lm[20];
  const pinkyPIP = lm[18];

  // Finger extended checks (y decreases going up in normalized coords)
  const indexUp = indexTip.y < indexPIP.y - 0.02; // stricter up
  
  // Others must be strictly curled down (tip below PIP)
  const middleDown = middleTip.y > middlePIP.y;
  const ringDown = ringTip.y > ringPIP.y;
  const pinkyDown = pinkyTip.y > pinkyPIP.y;
  
  // Open palm check (original relaxed conditions)
  const middleUp = middleTip.y < middlePIP.y;
  const ringUp = ringTip.y < ringPIP.y;
  const pinkyUp = pinkyTip.y < pinkyPIP.y;
  const thumbOut = Math.abs(thumbTip.x - thumbIP.x) > 0.03 || thumbTip.y < thumbIP.y;

  // Pinch detection: thumb tip close to index tip
  const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
  const isPinching = pinchDist < 0.06;

  // Gesture classification
  if (isPinching && !middleUp && !ringUp && !pinkyUp) {
    return 'pinch';
  }

  if (indexUp && middleUp && ringUp && pinkyUp && thumbOut) {
    return 'open_palm';
  }

  // Stricter drawing check: only index up, all others strictly down
  if (indexUp && middleDown && ringDown && pinkyDown) {
    return 'index_finger';
  }

  if (!indexUp && !middleUp && !ringUp && !pinkyUp) {
    return 'fist';
  }

  return 'idle';
}

// Stabilize gesture — require N consistent frames to switch
function stabilizeGesture(rawGesture) {
  // If same as current, stay put
  if (rawGesture === state.currentGesture) {
    state.previousGesture = rawGesture;
    state.gestureStableFrames = 0;
    return state.currentGesture;
  }

  // If this is the same candidate as last frame, count up
  if (rawGesture === state.previousGesture) {
    state.gestureStableFrames++;
  } else {
    // New candidate — reset counter
    state.previousGesture = rawGesture;
    state.gestureStableFrames = 1;
  }

  const threshold = rawGesture === 'pinch' ? 3 : 4;

  if (state.gestureStableFrames >= threshold) {
    const oldGesture = state.currentGesture;
    state.currentGesture = rawGesture;
    state.gestureStableFrames = 0;
    
    // Record start time of new gesture
    state.gestureStartTime = Date.now();

    if (oldGesture !== rawGesture) {
      onGestureChange(oldGesture, rawGesture);
    }
    return rawGesture;
  }

  return state.currentGesture;
}

function onGestureChange(from, to) {
  // Sound effects
  if (to === 'index_finger') playDrawStart();
  else if (to === 'open_palm') playModeSwitch();
  else if (to === 'pinch') playGrabSound();
  else if (from === 'index_finger') playDrawEnd();

  // End current stroke if we were drawing
  if (from === 'index_finger' && state.currentStroke) {
    if (state.currentStroke.points.length > 1) {
      state.strokes.push({ ...state.currentStroke });
    }
    state.currentStroke = null;
  }

  // End grab
  if (from === 'pinch') {
    endGrab();
  }

  // Update HUD
  updateGestureHUD(to);
}

function updateGestureHUD(gesture) {
  const map = {
    'index_finger': { icon: '☝️', label: 'Drawing', cls: 'drawing' },
    'open_palm':    { icon: '✋', label: 'Erasing', cls: 'erasing' },
    'pinch':        { icon: '🤏', label: 'Grab', cls: 'grabbing' },
    'fist':         { icon: '✊', label: 'Idle', cls: '' },
    'idle':         { icon: '🖐️', label: 'Ready', cls: '' },
    'none':         { icon: '👋', label: 'Show hand', cls: '' },
  };
  const info = map[gesture] || map['idle'];
  gestureIcon.textContent = info.icon;
  gestureLabel.textContent = info.label;
  gestureHud.className = info.cls;
}

// ── Drawing Logic ──────────────────────────────────────────────
function getLandmarkPos(landmark) {
  // Mirror X for natural feel, and scale to canvas
  return {
    x: (1 - landmark.x) * state.width,
    y: landmark.y * state.height
  };
}

function smoothPosition(rawPos) {
  state.smoothPos.x += (rawPos.x - state.smoothPos.x) * state.smoothFactor;
  state.smoothPos.y += (rawPos.y - state.smoothPos.y) * state.smoothFactor;
  return { x: state.smoothPos.x, y: state.smoothPos.y };
}

function handleDrawing(landmarks) {
  const indexTip = landmarks[8];
  const rawPos = getLandmarkPos(indexTip);
  const pos = smoothPosition(rawPos);

  // Buffer: Ignore drawing for first 300ms to avoid trailing lines from transition
  if (Date.now() - state.gestureStartTime < 300) {
    state.smoothPos = { ...rawPos };
    return;
  }

  if (!state.currentStroke) {
    state.currentStroke = {
      points: [pos],
      color: state.activeColor,
      thickness: state.thickness,
      glow: state.glowIntensity,
    };
    state.smoothPos = { ...rawPos };
  } else {
    state.currentStroke.points.push({ ...pos });
  }

  // Emit particles
  emitParticles(pos.x, pos.y, state.activeColor);
  
  redrawStrokes();
}

function handleErasing(landmarks) {
  const wrist = landmarks[0];
  const middleMCP = landmarks[9];
  // Palm center is roughly between wrist and middle MCP
  const palmCenter = {
    x: (1 - (wrist.x + middleMCP.x) / 2) * state.width,
    y: ((wrist.y + middleMCP.y) / 2) * state.height
  };

  const radius = state.eraserRadius;
  let erased = false;

  // Segment-based erasing: split strokes, keeping only points outside eraser
  const newStrokes = [];
  for (let i = 0; i < state.strokes.length; i++) {
    const stroke = state.strokes[i];
    const segments = [];
    let currentSegment = [];

    for (const p of stroke.points) {
      const dx = p.x - palmCenter.x;
      const dy = p.y - palmCenter.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist >= radius) {
        // Point is outside eraser — keep it
        currentSegment.push(p);
      } else {
        // Point is inside eraser — break the stroke here
        erased = true;
        if (currentSegment.length >= 2) {
          segments.push(currentSegment);
        }
        currentSegment = [];
      }
    }

    // Don't forget the last segment
    if (currentSegment.length >= 2) {
      segments.push(currentSegment);
    }

    // Convert segments back to strokes
    if (segments.length === 0 && stroke.points.length > 0) {
      // Entire stroke was erased — count as erased
      // (don't add anything back)
    } else if (segments.length === 1 && segments[0].length === stroke.points.length) {
      // Stroke was untouched
      newStrokes.push(stroke);
    } else {
      // Stroke was split into pieces
      for (const seg of segments) {
        newStrokes.push({
          points: seg,
          color: stroke.color,
          thickness: stroke.thickness,
          glow: stroke.glow,
        });
      }
    }
  }

  state.strokes = newStrokes;

  if (erased) {
    playEraseSound();
  }

  // Draw eraser circle on UI canvas
  uiCtx.beginPath();
  uiCtx.arc(palmCenter.x, palmCenter.y, radius, 0, Math.PI * 2);
  uiCtx.strokeStyle = 'rgba(255, 45, 107, 0.5)';
  uiCtx.lineWidth = 1.5;
  uiCtx.setLineDash([5, 5]);
  uiCtx.stroke();
  uiCtx.setLineDash([]);
  uiCtx.fillStyle = 'rgba(255, 45, 107, 0.05)';
  uiCtx.fill();

  redrawStrokes();
}

function handleGrab(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const pinchPos = {
    x: (1 - (thumbTip.x + indexTip.x) / 2) * state.width,
    y: ((thumbTip.y + indexTip.y) / 2) * state.height
  };

  if (!state.isGrabbing) {
    state.isGrabbing = true;
    state.grabStartPos = { ...pinchPos };
    state.nearestStrokeIdx = findNearestStroke(pinchPos);
  } else {
    const dx = pinchPos.x - state.grabStartPos.x;
    const dy = pinchPos.y - state.grabStartPos.y;

    if (state.nearestStrokeIdx >= 0 && state.nearestStrokeIdx < state.strokes.length) {
      const stroke = state.strokes[state.nearestStrokeIdx];
      const prevDx = state.grabOffset.x;
      const prevDy = state.grabOffset.y;
      const deltaDx = dx - prevDx;
      const deltaDy = dy - prevDy;
      
      for (let i = 0; i < stroke.points.length; i++) {
        stroke.points[i].x += deltaDx;
        stroke.points[i].y += deltaDy;
      }
    }

    state.grabOffset = { x: dx, y: dy };
  }

  // Draw grab indicator
  uiCtx.beginPath();
  uiCtx.arc(pinchPos.x, pinchPos.y, 18, 0, Math.PI * 2);
  uiCtx.strokeStyle = 'rgba(255, 215, 0, 0.7)';
  uiCtx.lineWidth = 2;
  uiCtx.stroke();
  uiCtx.fillStyle = 'rgba(255, 215, 0, 0.1)';
  uiCtx.fill();

  // Highlight grabbed stroke
  if (state.nearestStrokeIdx >= 0 && state.nearestStrokeIdx < state.strokes.length) {
    drawStrokeHighlight(state.strokes[state.nearestStrokeIdx]);
  }

  redrawStrokes();
}

function endGrab() {
  if (state.isGrabbing && state.nearestStrokeIdx >= 0) {
    playDropSound();
  }

  state.isGrabbing = false;
  state.grabStartPos = null;
  state.grabOffset = { x: 0, y: 0 };
  state.nearestStrokeIdx = -1;
  redrawStrokes();
}

function findNearestStroke(pos) {
  let minDist = Infinity;
  let nearestIdx = -1;

  for (let i = 0; i < state.strokes.length; i++) {
    const stroke = state.strokes[i];
    for (const p of stroke.points) {
      const d = Math.hypot(p.x - pos.x, p.y - pos.y);
      if (d < minDist) {
        minDist = d;
        nearestIdx = i;
      }
    }
  }

  return minDist < 80 ? nearestIdx : -1;
}



function drawStrokeHighlight(stroke) {
  if (!stroke || stroke.points.length < 2) return;
  uiCtx.save();
  uiCtx.beginPath();
  uiCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
  for (let i = 1; i < stroke.points.length; i++) {
    uiCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
  }
  uiCtx.strokeStyle = 'rgba(255, 215, 0, 0.3)';
  uiCtx.lineWidth = stroke.thickness + 12;
  uiCtx.lineCap = 'round';
  uiCtx.lineJoin = 'round';
  uiCtx.setLineDash([8, 8]);
  uiCtx.stroke();
  uiCtx.setLineDash([]);
  uiCtx.restore();
}

// ── Stroke Rendering with Glow ─────────────────────────────────
function drawGlowStroke(ctx, stroke, isCurrentStroke = false) {
  if (!stroke || stroke.points.length < 2) return;

  const pts = stroke.points;
  const color = stroke.color;
  const width = stroke.thickness;
  const glowMult = stroke.glow / 100;

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // Pass 1: Outer glow
  if (glowMult > 0) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i - 1];
      const curr = pts[i];
      const mx = (prev.x + curr.x) / 2;
      const my = (prev.y + curr.y) / 2;
      ctx.quadraticCurveTo(prev.x, prev.y, mx, my);
    }
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
    ctx.strokeStyle = color;
    ctx.lineWidth = width * 3;
    ctx.globalAlpha = 0.1 * glowMult;
    ctx.shadowColor = color;
    ctx.shadowBlur = 35 * glowMult;
    ctx.stroke();
  }

  // Pass 2: Mid glow
  if (glowMult > 0) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i - 1];
      const curr = pts[i];
      const mx = (prev.x + curr.x) / 2;
      const my = (prev.y + curr.y) / 2;
      ctx.quadraticCurveTo(prev.x, prev.y, mx, my);
    }
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
    ctx.strokeStyle = color;
    ctx.lineWidth = width * 1.6;
    ctx.globalAlpha = 0.35 * glowMult;
    ctx.shadowBlur = 15 * glowMult;
    ctx.stroke();
  }

  // Pass 3: Core line
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) {
    const prev = pts[i - 1];
    const curr = pts[i];
    const mx = (prev.x + curr.x) / 2;
    const my = (prev.y + curr.y) / 2;
    ctx.quadraticCurveTo(prev.x, prev.y, mx, my);
  }
  ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  ctx.strokeStyle = lightenColor(color, 0.5);
  ctx.lineWidth = width;
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 6 * glowMult;
  ctx.shadowColor = color;
  ctx.stroke();

  ctx.restore();
}

function lightenColor(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const nr = Math.min(255, Math.round(r + (255 - r) * amount));
  const ng = Math.min(255, Math.round(g + (255 - g) * amount));
  const nb = Math.min(255, Math.round(b + (255 - b) * amount));
  return `rgb(${nr}, ${ng}, ${nb})`;
}

function redrawStrokes() {
  drawingCtx.clearRect(0, 0, state.width, state.height);

  // Draw all completed strokes
  for (const stroke of state.strokes) {
    drawGlowStroke(drawingCtx, stroke);
  }

  // Draw current stroke
  if (state.currentStroke && state.currentStroke.points.length > 1) {
    drawGlowStroke(drawingCtx, state.currentStroke, true);
  }
}

// ── Particles ──────────────────────────────────────────────────
function emitParticles(x, y, color) {
  for (let i = 0; i < 2; i++) {
    state.particles.push({
      x, y,
      vx: (Math.random() - 0.5) * 3,
      vy: (Math.random() - 0.5) * 3,
      life: 1,
      decay: 0.02 + Math.random() * 0.03,
      size: 2 + Math.random() * 3,
      color,
    });
  }
}

function updateAndDrawParticles(ctx) {
  for (let i = state.particles.length - 1; i >= 0; i--) {
    const p = state.particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.life -= p.decay;
    p.size *= 0.97;

    if (p.life <= 0) {
      state.particles.splice(i, 1);
      continue;
    }

    ctx.save();
    ctx.globalAlpha = p.life * 0.7;
    ctx.fillStyle = p.color;
    ctx.shadowColor = p.color;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

// ── Hand Skeleton Drawing ──────────────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],      // Thumb
  [0,5],[5,6],[6,7],[7,8],      // Index
  [0,9],[9,10],[10,11],[11,12], // Middle  (via 0→9)
  [0,13],[13,14],[14,15],[15,16], // Ring  (via 0→13)
  [0,17],[17,18],[18,19],[19,20], // Pinky (via 0→17)
  [5,9],[9,13],[13,17],          // Palm connections
];

function drawHandSkeleton(ctx, landmarks) {
  if (!landmarks) return;

  ctx.save();
  ctx.globalAlpha = 0.3;

  // Draw connections
  for (const [a, b] of HAND_CONNECTIONS) {
    const pa = getLandmarkPos(landmarks[a]);
    const pb = getLandmarkPos(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Draw landmarks
  for (let i = 0; i < landmarks.length; i++) {
    const pos = getLandmarkPos(landmarks[i]);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.fill();
  }

  // Highlight fingertip
  const tips = [4, 8, 12, 16, 20];
  for (const t of tips) {
    const pos = getLandmarkPos(landmarks[t]);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.shadowColor = '#ffffff';
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  ctx.restore();
}

// ── Drawing cursor indicator ───────────────────────────────────
function drawCursorIndicator(ctx, landmarks, gesture) {
  if (gesture === 'index_finger') {
    const pos = getLandmarkPos(landmarks[8]);
    // Outer ring
    ctx.save();
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, state.thickness / 2 + 6, 0, Math.PI * 2);
    ctx.strokeStyle = state.activeColor;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.shadowColor = state.activeColor;
    ctx.shadowBlur = 8;
    ctx.stroke();
    // Inner dot
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = state.activeColor;
    ctx.globalAlpha = 0.9;
    ctx.fill();
    ctx.restore();
  }
}

// ── Main Render Loop ───────────────────────────────────────────
let lastVideoTime = -1;

function renderLoop() {
  if (!state.handLandmarker || !state.isReady) {
    requestAnimationFrame(renderLoop);
    return;
  }

  const video = webcamEl;
  const now = performance.now();

  // Draw camera feed
  cameraCtx.clearRect(0, 0, state.width, state.height);
  if (state.showCamera) {
    cameraCtx.save();
    cameraCtx.globalAlpha = state.cameraOpacity;
    // Mirror the camera
    cameraCtx.translate(state.width, 0);
    cameraCtx.scale(-1, 1);
    cameraCtx.drawImage(video, 0, 0, state.width, state.height);
    cameraCtx.restore();
  }

  // Clear UI overlay
  uiCtx.clearRect(0, 0, state.width, state.height);

  // Process hand landmarks
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const results = state.handLandmarker.detectForVideo(video, now);

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      const rawGesture = detectGesture(landmarks);
      const gesture = stabilizeGesture(rawGesture);

      if (!state.isModalOpen) {
        // Handle interactions
        if (gesture === 'index_finger') handleDrawing(landmarks);
        if (gesture === 'open_palm') handleErasing(landmarks);
        if (gesture === 'pinch') handleGrab(landmarks);
        
        // Finalize any in-progress stroke if not drawing
        if (gesture !== 'index_finger' && state.currentStroke && state.currentStroke.points.length > 1) {
          state.strokes.push({ ...state.currentStroke });
          state.currentStroke = null;
        }
      }

      // Render hand overlay
      drawHandSkeleton(uiCtx, landmarks);
      drawCursorIndicator(uiCtx, landmarks, gesture);
    } else {
      // No hand detected
      if (state.currentGesture !== 'none') {
        onGestureChange(state.currentGesture, 'none');
        state.currentGesture = 'none';
      }
      if (state.currentStroke && state.currentStroke.points.length > 1) {
        state.strokes.push({ ...state.currentStroke });
        state.currentStroke = null;
        redrawStrokes();
      }
    }
  }

  // Update particles
  updateAndDrawParticles(uiCtx);

  requestAnimationFrame(renderLoop);
}

// ── UI Event Handlers ──────────────────────────────────────────

// Color palette
document.querySelectorAll('.color-swatch').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.color-swatch').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.activeColor = btn.dataset.color;
    playTone(1000, 0.05, 'sine', 0.03);
  });
});

// Thickness
thicknessSlider.addEventListener('input', () => {
  state.thickness = parseInt(thicknessSlider.value);
  thicknessValue.textContent = `${state.thickness}px`;
});

// Glow
glowSlider.addEventListener('input', () => {
  state.glowIntensity = parseInt(glowSlider.value);
  glowValue.textContent = `${state.glowIntensity}%`;
});

// Undo
$('btn-undo').addEventListener('click', () => {
  if (state.strokes.length > 0) {
    state.strokes.pop();
    redrawStrokes();
    playTone(500, 0.08, 'sine', 0.03);
  }
});

// Clear
$('btn-clear').addEventListener('click', () => {
  state.strokes = [];
  state.currentStroke = null;
  state.particles = [];
  redrawStrokes();
  playTone(300, 0.15, 'triangle', 0.04);
});

// Camera toggle — cycles: Camera ON → Camera Dim → Dark Canvas
$('btn-camera-toggle').addEventListener('click', () => {
  if (state.showCamera && state.cameraOpacity > 0.2) {
    // Currently full → dim
    state.cameraOpacity = 0.15;
    cameraModeText.textContent = 'Camera DIM';
    cameraModeIndicator.classList.remove('dark-mode');
  } else if (state.showCamera && state.cameraOpacity <= 0.2) {
    // Currently dim → off
    state.showCamera = false;
    state.cameraOpacity = 0;
    cameraModeText.textContent = 'Dark Canvas';
    cameraModeIndicator.classList.add('dark-mode');
    $('btn-camera-toggle').classList.remove('active');
  } else {
    // Currently off → full
    state.showCamera = true;
    state.cameraOpacity = 0.35;
    cameraModeText.textContent = 'Camera ON';
    cameraModeIndicator.classList.remove('dark-mode');
    $('btn-camera-toggle').classList.add('active');
  }
  playModeSwitch();
});

// Also allow clicking the indicator to toggle
cameraModeIndicator.addEventListener('click', () => {
  $('btn-camera-toggle').click();
});

// Save
$('btn-save').addEventListener('click', () => {
  // Composite all canvases
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = state.width;
  exportCanvas.height = state.height;
  const exportCtx = exportCanvas.getContext('2d');

  // Dark background
  exportCtx.fillStyle = '#07070d';
  exportCtx.fillRect(0, 0, state.width, state.height);

  // Drawing layer
  exportCtx.drawImage(drawingCanvas, 0, 0);

  const link = document.createElement('a');
  link.download = `air-draw-${Date.now()}.png`;
  link.href = exportCanvas.toDataURL('image/png');
  link.click();

  playTone(800, 0.1, 'sine', 0.04);
});

// Onboarding start
btnStart.addEventListener('click', () => {
  onboardingModal.classList.add('hidden');
  state.isModalOpen = false;
  playTone(800, 0.1, 'sine', 0.04);
  
  // Reset HUD
  updateGestureHUD('idle');
});

// ── Initialization ─────────────────────────────────────────────
async function init() {
  resizeCanvases();

  try {
    // Load MediaPipe and webcam in parallel
    const [mpReady] = await Promise.all([
      initMediaPipe(),
      initWebcam()
    ]);

    state.isReady = true;

    // Complete the loader animation
    const loaderFill = document.querySelector('.loader-bar-fill');
    loaderFill.style.animation = 'none';
    loaderFill.style.width = '100%';
    loaderFill.style.transition = 'width 0.4s ease';

    // Fade out loading screen
    setTimeout(() => {
      loadingScreen.classList.add('fade-out');
      appEl.classList.remove('hidden');
      onboardingModal.classList.remove('hidden');
    }, 600);

    // Remove loading screen after fade
    setTimeout(() => {
      loadingScreen.style.display = 'none';
    }, 1200);

    // Start render loop
    renderLoop();

  } catch (error) {
    console.error('Failed to initialize Air Draw:', error);
    document.querySelector('.loader-subtitle').textContent = 
      'Error: Camera access required. Please allow camera permissions and reload.';
    document.querySelector('.loader-subtitle').style.color = '#ff2d6b';
    document.querySelector('.loader-bar').style.display = 'none';
  }
}

init();
