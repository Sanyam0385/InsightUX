const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const webview = document.getElementById('webview');
const addressBar = document.getElementById('address-bar');
const backBtn = document.getElementById('back-btn');
const forwardBtn = document.getElementById('forward-btn');
const reloadBtn = document.getElementById('reload-btn');
const homeBtn = document.getElementById('home-btn');

const startTrackBtn = document.getElementById('start-track-btn');
const gazeDot = document.getElementById('gaze-engine-status-dot');
const gazeText = document.getElementById('gaze-engine-status-text');
const trackDot = document.getElementById('tracking-status-dot');
const trackText = document.getElementById('tracking-status-text');

const HOME_URL = 'https://wikipedia.org';

// ─── One Euro Filter for Gaze Stabilization ───
class OneEuroFilter {
  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }

  alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(x, t) {
    if (this.xPrev === null) {
      this.xPrev = x;
      this.tPrev = t;
      return x;
    }

    let dt = (t - this.tPrev) / 1000.0; // in seconds
    if (dt <= 0) dt = 1e-3;
    this.tPrev = t;

    const dx = (x - this.xPrev) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1.0 - aD) * this.dxPrev;
    this.dxPrev = dxHat;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this.alpha(cutoff, dt);
    const xHat = a * x + (1.0 - a) * this.xPrev;
    this.xPrev = xHat;

    return xHat;
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }
}

const posterFilterX = new OneEuroFilter(0.35, 0.12);
const posterFilterY = new OneEuroFilter(0.35, 0.12);

// ─── Poster Analyzer State Variables ───
let currentMode = 'browser'; // 'browser' or 'poster'
let posterPath = '';
let posterElements = []; // { label, type, box: [x,y,w,h], score }
let posterGazeDwells = {}; // label -> ms
let posterMouseDwells = {}; // label -> ms
let posterClicksCount = 0;
let posterGazeCount = 0;
let posterMouseCount = 0;

let posterWS = null;
let posterCanvas = document.getElementById('poster-canvas');
let posterCtx = posterCanvas ? posterCanvas.getContext('2d') : null;
let posterImg = document.getElementById('poster-img');

let posterGazeX = null;
let posterGazeY = null;
let smoothGazeX = null;
let smoothGazeY = null;
let isPosterTracking = false;

let posterMouseX = 0;
let posterMouseY = 0;

let posterGazeTrail = [];
let posterMouseTrail = [];

let activePosterGazeElement = null;
let activePosterGazeStartTime = 0;
let activePosterMouseElement = null;
let activePosterMouseStartTime = 0;

let posterClicks = [];
let posterMouseDwellBuffer = [];
let posterGazeDwellBuffer = [];
let posterMouseTrailBuffer = [];
let posterGazeTrailBuffer = [];

let posterUIInterval = null;
let posterDwellDripInterval = null;

// Initialize the webview and load the default page
const preloadPath = 'file://' + path.join(__dirname, 'preload.js');
webview.setAttribute('preload', preloadPath);
webview.setAttribute('src', HOME_URL);

// Function to navigate the webview to the entered URL
function navigate() {
  let url = addressBar.value.trim();
  if (!url) return;

  // If the input doesn't look like a URL with a protocol, format it
  if (!/^https?:\/\//i.test(url)) {
    // If it doesn't contain a dot or has spaces (like a search query), search it on Google
    if (url.indexOf('.') === -1 || url.indexOf(' ') !== -1) {
      url = 'https://www.google.com/search?q=' + encodeURIComponent(url);
    } else {
      url = 'https://' + url;
    }
  }
  webview.src = url;
}

// Navigate on Enter keypress in address bar
addressBar.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    navigate();
    addressBar.blur();
  }
});

// Select all text in address bar when clicked/focused
addressBar.addEventListener('focus', () => {
  addressBar.select();
});

// Back button
backBtn.addEventListener('click', () => {
  if (webview.canGoBack()) {
    webview.goBack();
  }
});

// Forward button
forwardBtn.addEventListener('click', () => {
  if (webview.canGoForward()) {
    webview.goForward();
  }
});

// Reload button
reloadBtn.addEventListener('click', () => {
  webview.reload();
});

// Home button
homeBtn.addEventListener('click', () => {
  webview.src = HOME_URL;
});

// Synchronize address bar content as the webview navigates
webview.addEventListener('did-start-navigation', (e) => {
  addressBar.value = e.url;
});

webview.addEventListener('did-navigate', (e) => {
  addressBar.value = e.url;
});

webview.addEventListener('did-navigate-in-page', (e) => {
  addressBar.value = e.url;
});

// Update the main window title based on the loaded webview page title
webview.addEventListener('page-title-updated', (e) => {
  document.title = `${e.title} - WebGazer Analysis Browser`;
});

// ─── Gaze & Mouse Tracking Process Control & Logging ───
let pyProcess = null;
let isTracking = false;
let logsInterval = null;
let domLogPath = '';
let mouseLogPath = '';

function injectTrackingScript() {
  if (!isTracking) return;
  try {
    const trackingCodePath = path.join(__dirname, 'tracking_client.js');
    const trackingCode = fs.readFileSync(trackingCodePath, 'utf8');
    
    console.log('[Electron Shell] Injecting tracking_client.js into guest webview...');
    webview.executeJavaScript(trackingCode)
      .then(() => {
        console.log('[Electron Shell] Injected tracking_client.js successfully. Dispatching start event.');
        webview.executeJavaScript("window.dispatchEvent(new CustomEvent('insightux-start'))");
      })
      .catch(err => {
        console.error('[Electron Shell] Error executing tracking script inside webview:', err);
      });
  } catch (e) {
    console.error('[Electron Shell] Error reading tracking_client.js:', e);
  }
}

// Automatically re-inject and restart the tracking scripts on page navigation
webview.addEventListener('dom-ready', () => {
  if (isTracking) {
    console.log('[Electron Shell] Guest DOM ready, re-injecting tracking client...');
    injectTrackingScript();
  }
});

function startTracking() {
  if (isTracking) return;
  isTracking = true;

  startTrackBtn.classList.add('active');
  startTrackBtn.querySelector('.track-text').textContent = 'Stop Eye Tracking';
  
  trackDot.className = 'status-dot-indicator connected';
  trackText.textContent = 'Tracking Status: Active';

  gazeDot.className = 'status-dot-indicator disconnected';
  gazeText.textContent = 'Gaze Engine: Starting...';

  // Setup logging directory and file paths
  const sessionDir = path.join(__dirname, '..', 'sessions', 'live');
  if (!fs.existsSync(sessionDir)) {
    fs.mkdirSync(sessionDir, { recursive: true });
  }
  domLogPath = path.join(sessionDir, 'dom_log.jsonl');
  mouseLogPath = path.join(sessionDir, 'mouse_log.jsonl');

  // Clear previous session logs
  if (fs.existsSync(domLogPath)) {
    try { fs.unlinkSync(domLogPath); } catch(e) {}
  }
  if (fs.existsSync(mouseLogPath)) {
    try { fs.unlinkSync(mouseLogPath); } catch(e) {}
  }

  // Resolve Python executable path
  let pythonPath = 'python'; // Fallback to system python
  const localVenv = path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');
  
  if (fs.existsSync(localVenv)) {
    pythonPath = localVenv;
  }

  // Resolve Python script path (handling packaged app.asar.unpacked directory structure and local workspace paths)
  let scriptPath = '';
  const unpackedScriptPath = path.join(__dirname, 'app.asar.unpacked', 'python_backend', 'gaze_server.py');
  const unpackedScriptPath2 = path.join(__dirname, '..', 'app.asar.unpacked', 'python_backend', 'gaze_server.py');
  const devScriptPath = path.join(__dirname, 'python_backend', 'gaze_server.py');
  const rootScriptPath = path.join(__dirname, '..', 'gaze_server.py');
  
  if (fs.existsSync(unpackedScriptPath)) {
    scriptPath = unpackedScriptPath;
  } else if (fs.existsSync(unpackedScriptPath2)) {
    scriptPath = unpackedScriptPath2;
  } else if (fs.existsSync(devScriptPath)) {
    scriptPath = devScriptPath;
  } else if (fs.existsSync(rootScriptPath)) {
    scriptPath = rootScriptPath;
  } else {
    scriptPath = rootScriptPath; // Fallback to root path
  }

  const scriptDir = path.dirname(scriptPath);
  console.log('[Electron Shell] Spawning Python Gaze Server:');
  console.log(' - Python executable:', pythonPath);
  console.log(' - Script path:', scriptPath);
  console.log(' - Working directory (CWD):', scriptDir);
  console.log(' - Session Log Output Directory:', sessionDir);

  pyProcess = spawn(pythonPath, ['-u', scriptPath, sessionDir], { cwd: scriptDir });

  pyProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log('[Python Tracker]:', output);
    if (output.includes('WebSocket server listening')) {
      gazeDot.className = 'status-dot-indicator connected';
      gazeText.textContent = 'Gaze Engine: Online';
      
      // Inject tracking client and start session
      if (currentMode === 'browser') {
        injectTrackingScript();
      } else {
        startPosterTracking();
      }
    }
  });

  pyProcess.stderr.on('data', (data) => {
    console.error('[Python Error]:', data.toString());
  });

  pyProcess.on('close', (code) => {
    console.log('[Python Tracker] Process exited with code', code);
    pyProcess = null;
    stopTracking();
  });

  // Start periodic DOM and Mouse data retrieval loop (runs every 300ms)
  logsInterval = setInterval(async () => {
    if (!isTracking) return;

    if (currentMode === 'browser') {
      // 1. Get DOM elements visibility logs
      try {
        const rawAOIs = await webview.executeJavaScript("typeof window.insightuxAOIs === 'function' ? window.insightuxAOIs() : null");
        if (rawAOIs) {
          const rec = JSON.parse(rawAOIs);
          rec.type = "dom";
          rec.t = (Date.now() / 1000);
          fs.appendFileSync(domLogPath, JSON.stringify(rec) + '\n');
        }
      } catch (e) {
        // Ignored if webview hasn't loaded window.insightuxAOIs yet
      }

      // 2. Get Mouse details log (clicks, dwells, trails)
      try {
        const rawMouse = await webview.executeJavaScript("typeof window.insightuxMouseData === 'function' ? window.insightuxMouseData() : null");
        if (rawMouse) {
          const mouseData = JSON.parse(rawMouse);
          const tNow = (Date.now() / 1000);
          
          // Log clicks
          for (const click of mouseData.clicks || []) {
            click.type = "click";
            click.t = tNow;
            fs.appendFileSync(mouseLogPath, JSON.stringify(click) + '\n');
          }
          
          // Log mouse dwells
          for (const dwell of mouseData.dwells || []) {
            dwell.type = "mouse_dwell";
            dwell.t = tNow;
            fs.appendFileSync(mouseLogPath, JSON.stringify(dwell) + '\n');
          }
          
          // Log gaze dwells
          for (const gazeDwell of mouseData.gazeDwells || []) {
            gazeDwell.type = "gaze_dwell";
            gazeDwell.t = tNow;
            fs.appendFileSync(mouseLogPath, JSON.stringify(gazeDwell) + '\n');
          }

          // Log mouse trails
          if (mouseData.trail && mouseData.trail.length > 0) {
            fs.appendFileSync(mouseLogPath, JSON.stringify({
              type: "mouse_trail",
              t: tNow,
              points: mouseData.trail
            }) + '\n');
          }

          // Log gaze trails
          if (mouseData.gazeTrail && mouseData.gazeTrail.length > 0) {
            fs.appendFileSync(mouseLogPath, JSON.stringify({
              type: "gaze_trail",
              t: tNow,
              points: mouseData.gazeTrail
            }) + '\n');
          }
        }
      } catch (e) {
        // Ignored
      }
    } else {
      // ─── Poster Mode Local Telemetry Log Writer ───
      const tNow = (Date.now() / 1000);

      // 1. Log layout boxes (analogous to HTML DOM)
      if (posterElements.length > 0) {
        const rec = {
          url: "poster://" + path.basename(posterPath),
          scrollX: 0,
          scrollY: 0,
          viewport: { w: posterCanvas.width, h: posterCanvas.height },
          page: { w: posterCanvas.width, h: posterCanvas.height },
          aois: posterElements.map(el => {
            const [nx, ny, nw, nh] = el.box;
            return {
              label: el.label,
              x: Math.round(nx * posterCanvas.width),
              y: Math.round(ny * posterCanvas.height),
              w: Math.round(nw * posterCanvas.width),
              h: Math.round(nh * posterCanvas.height),
              sticky: true
            };
          }),
          type: "dom",
          t: tNow
        };
        fs.appendFileSync(domLogPath, JSON.stringify(rec) + '\n');
      }

      // 2. Log click events
      for (const click of posterClicks) {
        click.type = "click";
        click.t = tNow;
        fs.appendFileSync(mouseLogPath, JSON.stringify(click) + '\n');
      }
      posterClicks = [];

      // 3. Log mouse dwells
      for (const dwell of posterMouseDwellBuffer) {
        dwell.type = "mouse_dwell";
        dwell.t = tNow;
        fs.appendFileSync(mouseLogPath, JSON.stringify(dwell) + '\n');
      }
      posterMouseDwellBuffer = [];

      // 4. Log gaze dwells
      for (const gazeDwell of posterGazeDwellBuffer) {
        gazeDwell.type = "gaze_dwell";
        gazeDwell.t = tNow;
        fs.appendFileSync(mouseLogPath, JSON.stringify(gazeDwell) + '\n');
      }
      posterGazeDwellBuffer = [];

      // 5. Log mouse trails
      if (posterMouseTrailBuffer.length > 0) {
        fs.appendFileSync(mouseLogPath, JSON.stringify({
          type: "mouse_trail",
          t: tNow,
          points: posterMouseTrailBuffer
        }) + '\n');
        posterMouseTrailBuffer = [];
      }

      // 6. Log gaze trails
      if (posterGazeTrailBuffer.length > 0) {
        fs.appendFileSync(mouseLogPath, JSON.stringify({
          type: "gaze_trail",
          t: tNow,
          points: posterGazeTrailBuffer
        }) + '\n');
        posterGazeTrailBuffer = [];
      }
    }
  }, 300);
}

function stopTracking() {
  if (!isTracking) return;
  isTracking = false;

  startTrackBtn.classList.remove('active');
  startTrackBtn.querySelector('.track-text').textContent = 'Start Eye Tracking';

  trackDot.className = 'status-dot-indicator disconnected';
  trackText.textContent = 'Tracking Status: Idle';

  gazeDot.className = 'status-dot-indicator disconnected';
  gazeText.textContent = 'Gaze Engine: Offline';

  // Clear periodic log retrieval loop
  if (logsInterval) {
    clearInterval(logsInterval);
    logsInterval = null;
  }

  if (currentMode === 'browser') {
    // Signal guest webview to clean up tracking canvas and sidebars
    try {
      webview.executeJavaScript("window.dispatchEvent(new CustomEvent('insightux-stop'))");
    } catch (e) {
      // Ignore if page is already closed or navigated away
    }
  } else {
    stopPosterTracking();
  }

  if (pyProcess) {
    try {
      pyProcess.kill();
    } catch (e) {
      console.error('[Electron Shell] Error killing Python process:', e);
    }
    pyProcess = null;
  }
}

// ─── Poster Mode Setup & Tracking Methods ───

function startPosterTracking() {
  if (isPosterTracking) return;
  isPosterTracking = true;

  console.log('[Electron Shell] Starting Poster Mode tracking...');

  // Reset counters and buffers
  posterClicks = [];
  posterMouseDwellBuffer = [];
  posterGazeDwellBuffer = [];
  posterMouseTrailBuffer = [];
  posterGazeTrailBuffer = [];

  posterGazeDwells = {};
  posterMouseDwells = {};
  posterClicksCount = 0;
  posterGazeCount = 0;
  posterMouseCount = 0;

  posterGazeTrail = [];
  posterMouseTrail = [];
  smoothGazeX = null;
  smoothGazeY = null;
  
  posterFilterX.reset();
  posterFilterY.reset();

  activePosterGazeElement = null;
  activePosterGazeStartTime = Date.now();
  activePosterMouseElement = null;
  activePosterMouseStartTime = Date.now();

  // Connect websocket for coordinates
  connectPosterGripWS(); // Helper maps to connectPosterGazeWS below

  // Resize canvas overlay
  resizePosterCanvas();

  // Render elements in sidebar
  renderPosterElementsList();

  // Start sidebar updating loop
  posterUIInterval = setInterval(() => {
    if (isPosterTracking) {
      renderPosterElementsList();
    }
  }, 500);

  // Start continuous dwell drip
  posterDwellDripInterval = setInterval(() => {
    if (!isPosterTracking) return;
    const now = Date.now();

    // Accumulate gaze dwell
    if (activePosterGazeElement) {
      const duration = now - activePosterGazeStartTime;
      if (duration > 50) {
        posterGazeDwells[activePosterGazeElement] = (posterGazeDwells[activePosterGazeElement] || 0) + duration;
        posterGazeDwellBuffer.push({ element: activePosterGazeElement, duration: duration });
        activePosterGazeStartTime = now;
      }
    }

    // Accumulate mouse dwell
    if (activePosterMouseElement) {
      const duration = now - activePosterMouseStartTime;
      if (duration > 50) {
        posterMouseDwells[activePosterMouseElement] = (posterMouseDwells[activePosterMouseElement] || 0) + duration;
        posterMouseDwellBuffer.push({ element: activePosterMouseElement, duration: duration });
        activePosterMouseStartTime = now;
      }
    }
  }, 100);

  // Start animation loop
  requestAnimationFrame(drawPosterOverlay);
}

function stopPosterTracking() {
  if (!isPosterTracking) return;
  isPosterTracking = false;

  console.log('[Electron Shell] Stopping Poster Mode tracking...');

  if (posterWS) {
    try { posterWS.close(); } catch(e) {}
    posterWS = null;
  }

  if (posterUIInterval) {
    clearInterval(posterUIInterval);
    posterUIInterval = null;
  }

  if (posterDwellDripInterval) {
    clearInterval(posterDwellDripInterval);
    posterDwellDripInterval = null;
  }

  const gazeLbl = document.getElementById('poster-gaze-lbl');
  if (gazeLbl) gazeLbl.textContent = '-';
  const mouseLbl = document.getElementById('poster-mouse-lbl');
  if (mouseLbl) mouseLbl.textContent = '-';

  // Render list final state
  renderPosterElementsList();

  // Clear canvas
  if (posterCtx && posterCanvas) {
    posterCtx.clearRect(0, 0, posterCanvas.width, posterCanvas.height);
  }
}

function connectPosterGazeWS() {
  if (!isPosterTracking) return;

  console.log('[Electron Shell] Connecting to Gaze Python WS server...');
  posterWS = new WebSocket("ws://localhost:8765");

  posterWS.onopen = () => {
    console.log('[Electron Shell] Connected to gaze WS server.');
  };

  posterWS.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "gaze") {
        posterGazeX = data.x;
        posterGazeY = data.y;
        posterGazeCount++;

        processPosterGaze(data.x, data.y);
      }
    } catch (e) {
      console.error('[Electron Shell] Error parsing WS message:', e);
    }
  };

  posterWS.onclose = () => {
    console.log('[Electron Shell] Gaze WS closed.');
    if (isPosterTracking) {
      setTimeout(connectPosterGazeWS, 1500); // Reconnect
    }
  };
}

// Redirect connectPosterGripWS helper
function connectPosterGripWS() {
  connectPosterGazeWS();
}

function processPosterGaze(gx, gy) {
  if (!posterCanvas || !posterImg) return;

  // Map screen coordinates using percentage-based scaling consistent with website tracking
  const sw = window.screen.width;
  const sh = window.screen.height;
  const fx = gx / sw;
  const fy = gy / sh;

  // Translate percentages to window viewport CSS coordinates
  const vx = fx * window.innerWidth;
  const vy = fy * window.innerHeight;

  // Get canvas boundaries relative to viewport
  const rect = posterCanvas.getBoundingClientRect();
  const cx = vx - rect.left;
  const cy = vy - rect.top;

  // Check boundaries
  if (cx >= 0 && cx <= rect.width && cy >= 0 && cy <= rect.height) {
    const now = Date.now();
    
    // Smooth using One Euro filter (replaces raw EMA to fix jitter/vibration)
    smoothGazeX = posterFilterX.filter(cx, now);
    smoothGazeY = posterFilterY.filter(cy, now);

    const nx = cx / rect.width;
    const ny = cy / rect.height;

    // Buffer for logs and paths
    posterGazeTrailBuffer.push({ x: Math.round(cx), y: Math.round(cy) });
    posterGazeTrail.push({ x: cx, y: cy });
    if (posterGazeTrail.length > 80) posterGazeTrail.shift();

    // Collision detection
    let matched = null;
    for (const el of posterElements) {
      const [bx, by, bw, bh] = el.box;
      if (nx >= bx && nx <= bx + bw && ny >= by && ny <= by + bh) {
        if (!matched || (bw * bh) < (matched.box[2] * matched.box[3])) {
          matched = el;
        }
      }
    }

    const activeLabel = matched ? matched.label : null;

    const gazeLbl = document.getElementById('poster-gaze-lbl');
    if (gazeLbl) gazeLbl.textContent = activeLabel || '-';

    if (activeLabel !== activePosterGazeElement) {
      if (activePosterGazeElement) {
        const duration = now - activePosterGazeStartTime;
        if (duration > 50) {
          posterGazeDwells[activePosterGazeElement] = (posterGazeDwells[activePosterGazeElement] || 0) + duration;
          posterGazeDwellBuffer.push({ element: activePosterGazeElement, duration: duration });
        }
      }
      activePosterGazeElement = activeLabel;
      activePosterGazeStartTime = now;
    }
  } else {
    // Left boundary
    const now = Date.now();
    if (activePosterGazeElement) {
      const duration = now - activePosterGazeStartTime;
      if (duration > 50) {
        posterGazeDwells[activePosterGazeElement] = (posterGazeDwells[activePosterGazeElement] || 0) + duration;
        posterGazeDwellBuffer.push({ element: activePosterGazeElement, duration: duration });
      }
      activePosterGazeElement = null;
      const gazeLbl = document.getElementById('poster-gaze-lbl');
      if (gazeLbl) gazeLbl.textContent = '-';
    }
  }
}

function resizePosterCanvas() {
  if (!posterCanvas || !posterImg) return;
  posterCanvas.width = posterImg.clientWidth;
  posterCanvas.height = posterImg.clientHeight;
}

function renderPosterElementsList() {
  const container = document.getElementById('poster-elements-list');
  if (!container) return;

  if (posterElements.length === 0) {
    container.innerHTML = '<p style="font-size: 11px; color: var(--text-muted); text-align: center; margin: 20px 0;">No poster loaded yet.</p>';
    return;
  }

  // Sort elements by gaze dwell time descending
  const sorted = [...posterElements].sort((a, b) => {
    const dwellA = posterGazeDwells[a.label] || 0;
    const dwellB = posterGazeDwells[b.label] || 0;
    return dwellB - dwellA;
  });

  const maxDwell = Math.max(...sorted.map(el => posterGazeDwells[el.label] || 0), 1);

  let html = '';
  sorted.forEach(el => {
    const dwell = posterGazeDwells[el.label] || 0;
    const pct = (dwell / maxDwell) * 100;
    const isMostLooked = (dwell > 0 && el.label === sorted[0].label);

    html += `
      <div class="poster-el-item" style="${isMostLooked ? 'border-color: rgba(255, 45, 240, 0.4); background: rgba(123, 47, 190, 0.1);' : ''}">
        <div class="poster-el-meta">
          <span class="poster-el-name" title="${el.label}">${isMostLooked ? '👑 ' : ''}${el.label}</span>
          <span class="poster-el-time">${(dwell / 1000).toFixed(1)}s</span>
        </div>
        <div class="poster-progress-bg">
          <div class="poster-progress-fill" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

function drawPosterOverlay() {
  if (!isPosterTracking || !posterCanvas || !posterCtx) return;

  const w = posterCanvas.width;
  const h = posterCanvas.height;

  posterCtx.clearRect(0, 0, w, h);

  // 1. Draw detected bounding boxes
  posterElements.forEach(el => {
    const [bx, by, bw, bh] = el.box;
    const px = bx * w;
    const py = by * h;
    const pw = bw * w;
    const ph = bh * h;

    const isGazeActive = (el.label === activePosterGazeElement);
    const isMouseActive = (el.label === activePosterMouseElement);

    posterCtx.save();
    if (isGazeActive) {
      posterCtx.strokeStyle = '#FF2DF0';
      posterCtx.lineWidth = 3;
      posterCtx.fillStyle = 'rgba(255, 45, 240, 0.1)';
      posterCtx.fillRect(px, py, pw, ph);
    } else if (isMouseActive) {
      posterCtx.strokeStyle = '#00BFFF';
      posterCtx.lineWidth = 2;
      posterCtx.fillStyle = 'rgba(0, 191, 255, 0.05)';
      posterCtx.fillRect(px, py, pw, ph);
    } else {
      posterCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      posterCtx.lineWidth = 1;
    }

    posterCtx.strokeRect(px, py, pw, ph);

    if (isGazeActive || isMouseActive) {
      posterCtx.font = 'bold 10px sans-serif';
      const labelText = el.label;
      const tw = posterCtx.measureText(labelText).width + 8;
      const ly = Math.max(0, py - 14);

      posterCtx.fillStyle = isGazeActive ? '#FF2DF0' : '#00BFFF';
      posterCtx.fillRect(px, ly, tw, 13);

      posterCtx.fillStyle = '#FFFFFF';
      posterCtx.fillText(labelText, px + 4, ly + 10);
    }
    posterCtx.restore();
  });

  // 2. Draw Trails
  // Mouse
  if (posterMouseTrail.length > 1) {
    posterCtx.save();
    posterCtx.beginPath();
    posterCtx.strokeStyle = 'rgba(30, 144, 255, 0.4)';
    posterCtx.lineWidth = 2;
    posterCtx.lineJoin = 'round';
    posterCtx.lineCap = 'round';
    posterMouseTrail.forEach((p, idx) => {
      if (idx === 0) posterCtx.moveTo(p.x, p.y);
      else posterCtx.lineTo(p.x, p.y);
    });
    posterCtx.stroke();
    posterCtx.restore();
  }

  // Gaze
  if (posterGazeTrail.length > 1) {
    posterCtx.save();
    posterCtx.beginPath();
    posterCtx.strokeStyle = 'rgba(255, 69, 0, 0.45)';
    posterCtx.lineWidth = 2.5;
    posterCtx.lineJoin = 'round';
    posterCtx.lineCap = 'round';
    posterGazeTrail.forEach((p, idx) => {
      if (idx === 0) posterCtx.moveTo(p.x, p.y);
      else posterCtx.lineTo(p.x, p.y);
    });
    posterCtx.stroke();
    posterCtx.restore();
  }

  // 3. Draw Gaze Dot
  if (smoothGazeX !== null) {
    posterCtx.save();
    posterCtx.beginPath();
    posterCtx.arc(smoothGazeX, smoothGazeY, 12, 0, 2 * Math.PI);
    posterCtx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    posterCtx.fill();

    posterCtx.beginPath();
    posterCtx.arc(smoothGazeX, smoothGazeY, 8, 0, 2 * Math.PI);
    posterCtx.fillStyle = '#FF2DF0';
    posterCtx.fill();

    posterCtx.lineWidth = 1.5;
    posterCtx.strokeStyle = 'rgba(0,0,0,0.5)';
    posterCtx.stroke();
    posterCtx.restore();
  }

  // 4. Draw Mouse Dot
  if (posterMouseX !== 0) {
    posterCtx.save();
    posterCtx.beginPath();
    posterCtx.arc(posterMouseX, posterMouseY, 6, 0, 2 * Math.PI);
    posterCtx.fillStyle = '#00BFFF';
    posterCtx.fill();

    posterCtx.lineWidth = 1;
    posterCtx.strokeStyle = 'rgba(255,255,255,0.8)';
    posterCtx.stroke();
    posterCtx.restore();
  }

  requestAnimationFrame(drawPosterOverlay);
}

function setupPosterMouseListeners() {
  if (!posterCanvas) return;

  posterCanvas.addEventListener('mousemove', (e) => {
    if (!isPosterTracking) return;

    const rect = posterCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    posterMouseX = cx;
    posterMouseY = cy;
    posterMouseCount++;

    posterMouseTrail.push({ x: cx, y: cy });
    if (posterMouseTrail.length > 80) posterMouseTrail.shift();
    posterMouseTrailBuffer.push({ x: Math.round(cx), y: Math.round(cy) });

    const nx = cx / rect.width;
    const ny = cy / rect.height;

    let matched = null;
    for (const el of posterElements) {
      const [bx, by, bw, bh] = el.box;
      if (nx >= bx && nx <= bx + bw && ny >= by && ny <= by + bh) {
        if (!matched || (bw * bh) < (matched.box[2] * matched.box[3])) {
          matched = el;
        }
      }
    }

    const now = Date.now();
    const activeLabel = matched ? matched.label : null;

    const mouseLbl = document.getElementById('poster-mouse-lbl');
    if (mouseLbl) mouseLbl.textContent = activeLabel || '-';

    if (activeLabel !== activePosterMouseElement) {
      if (activePosterMouseElement) {
        const duration = now - activePosterMouseStartTime;
        if (duration > 50) {
          posterMouseDwells[activePosterMouseElement] = (posterMouseDwells[activePosterMouseElement] || 0) + duration;
          posterMouseDwellBuffer.push({ element: activePosterMouseElement, duration: duration });
        }
      }
      activePosterMouseElement = activeLabel;
      activePosterMouseStartTime = now;
    }
  });

  posterCanvas.addEventListener('mouseleave', () => {
    const now = Date.now();
    if (activePosterMouseElement) {
      const duration = now - activePosterMouseStartTime;
      if (duration > 50) {
        posterMouseDwells[activePosterMouseElement] = (posterMouseDwells[activePosterMouseElement] || 0) + duration;
        posterMouseDwellBuffer.push({ element: activePosterMouseElement, duration: duration });
      }
      activePosterMouseElement = null;
      const mouseLbl = document.getElementById('poster-mouse-lbl');
      if (mouseLbl) mouseLbl.textContent = '-';
    }
    posterMouseX = 0;
  });

  posterCanvas.addEventListener('click', (e) => {
    if (!isPosterTracking) return;

    const rect = posterCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const nx = cx / rect.width;
    const ny = cy / rect.height;

    let matched = null;
    for (const el of posterElements) {
      const [bx, by, bw, bh] = el.box;
      if (nx >= bx && nx <= bx + bw && ny >= by && ny <= by + bh) {
        if (!matched || (bw * bh) < (matched.box[2] * matched.box[3])) {
          matched = el;
        }
      }
    }

    const label = matched ? matched.label : 'Background';

    const clickEvent = {
      timestamp: new Date().toLocaleTimeString(),
      x: Math.round(cx),
      y: Math.round(cy),
      gazeX: smoothGazeX !== null ? Math.round(smoothGazeX) : null,
      gazeY: smoothGazeY !== null ? Math.round(smoothGazeY) : null,
      element: label,
      text: label,
      url: "poster://" + path.basename(posterPath)
    };

    posterClicks.push(clickEvent);
    posterClicksCount++;
  });
}

// ─── Mode Switching and Upload Event Listeners ───

const modeBrowserBtn = document.getElementById('mode-browser-btn');
const modePosterBtn = document.getElementById('mode-poster-btn');
const posterView = document.getElementById('poster-view');
const posterStatsSection = document.getElementById('poster-stats-section');
const cameraFeedbackSection = document.getElementById('camera-feedback-section');

const uploadZone = document.getElementById('poster-upload-zone');
const fileInput = document.getElementById('poster-file-input');
const workspace = document.getElementById('poster-workspace');
const imgWrapper = document.getElementById('poster-img-wrapper');
const changePosterBtn = document.getElementById('change-poster-btn');

modeBrowserBtn.addEventListener('click', () => {
  if (currentMode === 'browser') return;
  if (isTracking) stopTracking();

  currentMode = 'browser';
  modeBrowserBtn.classList.add('active');
  modePosterBtn.classList.remove('active');

  webview.style.display = 'flex';
  posterView.style.display = 'none';
  posterStatsSection.style.display = 'none';
  addressBar.style.display = 'block';
  
  addressBar.value = webview.src || HOME_URL;
});

modePosterBtn.addEventListener('click', () => {
  if (currentMode === 'poster') return;
  if (isTracking) stopTracking();

  currentMode = 'poster';
  modePosterBtn.classList.add('active');
  modeBrowserBtn.classList.remove('active');

  webview.style.display = 'none';
  posterView.style.display = 'flex';
  posterStatsSection.style.display = 'block';
  addressBar.style.display = 'none';

  resizePosterCanvas();
});

uploadZone.addEventListener('click', () => {
  fileInput.click();
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    loadPosterImage(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    loadPosterImage(e.target.files[0]);
  }
});

if (changePosterBtn) {
  changePosterBtn.addEventListener('click', () => {
    if (isTracking) stopTracking();
    resetUploadZone();
  });
}

function loadPosterImage(file) {
  posterPath = file.path;

  // Show loading
  uploadZone.querySelector('h2').textContent = 'Analyzing Poster...';
  uploadZone.querySelector('p').textContent = 'Running hybrid layout and object detector...';

  posterImg.src = 'file://' + posterPath;

  let pythonPath = 'python';
  const localVenv = path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');
  if (fs.existsSync(localVenv)) pythonPath = localVenv;

  // Spawning local analyzer
  const rootAnalyzerScript = path.join(__dirname, '..', 'poster_analyzer.py');
  const devAnalyzerScript = path.join(__dirname, 'poster_analyzer.py');
  const unpackedAnalyzerScript = path.join(__dirname, 'app.asar.unpacked', 'python_backend', 'poster_analyzer.py');
  
  let analyzerScript = rootAnalyzerScript;
  if (fs.existsSync(rootAnalyzerScript)) {
    analyzerScript = rootAnalyzerScript;
  } else if (fs.existsSync(devAnalyzerScript)) {
    analyzerScript = devAnalyzerScript;
  } else if (fs.existsSync(unpackedAnalyzerScript)) {
    analyzerScript = unpackedAnalyzerScript;
  }

  console.log('[Electron Shell] Running image analysis on:', posterPath);
  const child = spawn(pythonPath, [analyzerScript, posterPath]);

  let outputData = '';
  child.stdout.on('data', (data) => {
    outputData += data.toString();
  });

  child.stderr.on('data', (data) => {
    console.error('[Analyzer Error]:', data.toString());
  });

  child.on('close', (code) => {
    console.log('[Analyzer Process] exited with code', code);
    try {
      const jsonStart = outputData.indexOf('{');
      if (jsonStart === -1) {
        throw new Error('No JSON output found from analyzer. Output received:\n' + outputData);
      }
      const rawJson = outputData.substring(jsonStart);
      const result = JSON.parse(rawJson);
      if (result.success) {
        posterElements = result.elements;
        console.log('[Analyzer] Loaded elements:', posterElements);

        uploadZone.style.display = 'none';
        workspace.style.display = 'flex';
        imgWrapper.style.display = 'inline-block';

        posterImg.onload = () => {
          resizePosterCanvas();
          renderPosterElementsList();
        };
      } else {
        alert('Analysis failed: ' + result.error);
        resetUploadZone();
      }
    } catch (err) {
      console.error('[Analyzer Parse Error]:', err);
      alert('Failed to parse image analyzer output.');
      resetUploadZone();
    }
  });
}

function resetUploadZone() {
  uploadZone.style.display = 'flex';
  uploadZone.querySelector('h2').textContent = 'Upload Poster Image';
  uploadZone.querySelector('p').textContent = 'Drag & drop or click to select a poster image';
  workspace.style.display = 'none';
  imgWrapper.style.display = 'none';
  posterPath = '';
  posterElements = [];
  renderPosterElementsList();
}

// Register mouse event handlers on canvas
setupPosterMouseListeners();

// ─── Click & Control Listeners ───

startTrackBtn.addEventListener('click', () => {
  if (isTracking) {
    stopTracking();
  } else {
    startTracking();
  }
});

// Kill process on close/exit
window.addEventListener('beforeunload', () => {
  if (pyProcess) {
    pyProcess.kill();
  }
});
