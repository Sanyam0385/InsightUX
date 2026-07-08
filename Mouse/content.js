console.log("Mouse Tracker content script loaded.");

let isTracking = false;
let canvas = null;
let gazeCursor = null;
let showGazeCursor = true;
let heatmapType = "combined";

// Buffers (Mouse)
let trailBuffer = [];   // For visual path
let heatmapBuffer = []; // For intensity (dwell)
let clickBuffer = [];
let dwellBuffer = []; // For element interest

// Buffers (Gaze)
let gazeTrailBuffer = [];
let gazeHeatmapBuffer = [];
let gazeDwellBuffer = [];

// Sampling configuration (Mouse)
let lastClientX = 0, lastClientY = 0; // Viewport coordinates for element detection
let lastPageX = 0, lastPageY = 0;     // Document coordinates for heatmap/trail
let lastSampleTime = 0;
const sampleRate = 50; // Sample every 50ms (~20fps)
const velocityThreshold = 1.5; // pixels per ms
const dwellSampleRate = 1000; // Check dwell every 1s

// Sampling configuration (Gaze)
let lastGazeClientX = 0, lastGazeClientY = 0;
let lastGazePageX = 0, lastGazePageY = 0;
let lastGazeSampleTime = 0;

// Heatmap Logic Customization (Mouse)
let stationaryStart = 0;
let isStationary = false;
const DWELL_THRESHOLD = 5000; // 5 seconds dwell required for heatmap

// Heatmap Logic Customization (Gaze)
let gazeStationaryStart = 0;
let isGazeStationary = false;

// Initialize
chrome.runtime.sendMessage({ action: "getStatus" }, (response) => {
    if (response) {
        isTracking = response.isTracking;
        showGazeCursor = response.showGazeCursor !== false;
        heatmapType = response.heatmapType || "combined";
    }
});

// Main Tracking Loop (Time-based for both Mouse and Gaze)
setInterval(() => {
    if (!isTracking) return;

    const currentScrollX = window.scrollX;
    const currentScrollY = window.scrollY;
    const now = Date.now();

    // ─── 1. MOUSE TRACKING ───
    if (lastClientX !== 0 || lastPageX !== 0) {
        if (lastPageX === 0 && lastPageY === 0 && lastClientX !== 0) {
            lastPageX = lastClientX + currentScrollX;
            lastPageY = lastClientY + currentScrollY;
        }

        const currentPageX = (lastClientX !== 0) ? (lastClientX + currentScrollX) : lastPageX;
        const currentPageY = (lastClientY !== 0) ? (lastClientY + currentScrollY) : lastPageY;

        if (currentPageX !== 0 || currentPageY !== 0) {
            const dt = now - lastSampleTime;
            if (dt <= 1000) {
                const dx = currentPageX - lastPageX;
                const dy = currentPageY - lastPageY;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance > 2) {
                    trailBuffer.push({ x: currentPageX, y: currentPageY });
                    lastPageX = currentPageX;
                    lastPageY = currentPageY;
                    isStationary = false;
                    stationaryStart = 0;
                } else {
                    if (!isStationary) {
                        isStationary = true;
                        stationaryStart = now;
                    } else {
                        const dwellDuration = now - stationaryStart;
                        if (dwellDuration > DWELL_THRESHOLD) {
                            heatmapBuffer.push({ x: currentPageX, y: currentPageY });
                        }
                    }
                }
            }
            lastSampleTime = now;
        }
    }

    // ─── 2. GAZE TRACKING ───
    if (lastGazeClientX !== 0) {
        if (lastGazePageX === 0 && lastGazePageY === 0) {
            lastGazePageX = lastGazeClientX + currentScrollX;
            lastGazePageY = lastGazeClientY + currentScrollY;
        }

        const currentGazePageX = lastGazeClientX + currentScrollX;
        const currentGazePageY = lastGazeClientY + currentScrollY;

        const dtGaze = now - lastGazeSampleTime;
        if (dtGaze <= 1000) {
            const dx = currentGazePageX - lastGazePageX;
            const dy = currentGazePageY - lastGazePageY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            // Gaze has natural micro-tremor, use a slightly larger threshold (e.g. 4px)
            if (distance > 4) {
                gazeTrailBuffer.push({ x: currentGazePageX, y: currentGazePageY });
                lastGazePageX = currentGazePageX;
                lastGazePageY = currentGazePageY;
                isGazeStationary = false;
                gazeStationaryStart = 0;
            } else {
                if (!isGazeStationary) {
                    isGazeStationary = true;
                    gazeStationaryStart = now;
                } else {
                    const dwellDuration = now - gazeStationaryStart;
                    if (dwellDuration > DWELL_THRESHOLD) {
                        gazeHeatmapBuffer.push({ x: currentGazePageX, y: currentGazePageY });
                    }
                }
            }
        }
        lastGazeSampleTime = now;
    }
}, sampleRate);

let lastPageXStored = 0, lastPageYStored = 0;

// Sync Loop
const syncInterval = 2000;
setInterval(() => {
    if (!isTracking) return;

    // Flush ongoing hover so we have real-time updates while stationary
    const now = Date.now();
    
    // Mouse hover
    if (currentHoverLabel && currentHoverStartTime > 0) {
        const duration = now - currentHoverStartTime;
        if (duration > 50) {
            dwellBuffer.push({ element: currentHoverLabel, duration: duration });
            currentHoverStartTime = now; // Reset start time
        }
    }

    // Gaze hover
    if (currentGazeHoverLabel && currentGazeHoverStartTime > 0) {
        const durationGaze = now - currentGazeHoverStartTime;
        if (durationGaze > 50) {
            gazeDwellBuffer.push({ element: currentGazeHoverLabel, duration: durationGaze });
            currentGazeHoverStartTime = now;
        }
    }

    if (trailBuffer.length > 0 || heatmapBuffer.length > 0 || clickBuffer.length > 0 || dwellBuffer.length > 0 ||
        gazeTrailBuffer.length > 0 || gazeHeatmapBuffer.length > 0 || gazeDwellBuffer.length > 0) {
        
        let dwellToSend = null;
        if (dwellBuffer.length > 0) {
            dwellToSend = [...dwellBuffer];
            dwellBuffer = [];
        }

        let gazeDwellToSend = null;
        if (gazeDwellBuffer.length > 0) {
            gazeDwellToSend = [...gazeDwellBuffer];
            gazeDwellBuffer = [];
        }

        chrome.runtime.sendMessage({
            action: "logData",
            trail: trailBuffer.length > 0 ? trailBuffer : null,
            heatmap: heatmapBuffer.length > 0 ? heatmapBuffer : null,
            click: clickBuffer.length > 0 ? clickBuffer : null,
            dwell: dwellToSend,
            
            gazeTrail: gazeTrailBuffer.length > 0 ? gazeTrailBuffer : null,
            gazeHeatmap: gazeHeatmapBuffer.length > 0 ? gazeHeatmapBuffer : null,
            gazeDwell: gazeDwellToSend
        });

        trailBuffer = [];
        heatmapBuffer = [];
        clickBuffer = [];
        gazeTrailBuffer = [];
        gazeHeatmapBuffer = [];
    }
}, syncInterval);

// Accurate Dwell Tracking using Event Listeners instead of polling (Mouse)
let currentHoverLabel = null;
let currentHoverStartTime = 0;

function handleHoverChange(newLabel) {
    if (!isTracking) return;
    if (newLabel !== currentHoverLabel) {
        const now = Date.now();
        if (currentHoverLabel && currentHoverStartTime > 0) {
            const duration = now - currentHoverStartTime;
            if (duration > 10) { // Small threshold
                dwellBuffer.push({ element: currentHoverLabel, duration: duration });
            }
        }
        currentHoverLabel = newLabel;
        currentHoverStartTime = newLabel ? now : 0;
    }
}

// Accurate Gaze Dwell Tracking (Gaze)
let currentGazeHoverLabel = null;
let currentGazeHoverStartTime = 0;

function handleGazeHoverChange(newLabel) {
    if (!isTracking) return;
    if (newLabel !== currentGazeHoverLabel) {
        const now = Date.now();
        if (currentGazeHoverLabel && currentGazeHoverStartTime > 0) {
            const duration = now - currentGazeHoverStartTime;
            if (duration > 10) { // Small threshold
                gazeDwellBuffer.push({ element: currentGazeHoverLabel, duration: duration });
            }
        }
        currentGazeHoverLabel = newLabel;
        currentGazeHoverStartTime = newLabel ? now : 0;
    }
}

// Update eye gaze cursor on the page
function updateGazeCursor(px, py, visible) {
    if (!visible || !isTracking) {
        if (gazeCursor) {
            gazeCursor.style.display = 'none';
        }
        return;
    }

    if (!gazeCursor) {
        gazeCursor = document.createElement('div');
        gazeCursor.id = 'insightux-gaze-cursor';
        gazeCursor.style.position = 'absolute';
        gazeCursor.style.width = '24px';
        gazeCursor.style.height = '24px';
        gazeCursor.style.borderRadius = '50%';
        gazeCursor.style.backgroundColor = 'rgba(255, 69, 0, 0.4)'; // Transparent orange-red
        gazeCursor.style.border = '2px solid rgba(255, 69, 0, 0.8)';
        gazeCursor.style.pointerEvents = 'none'; // Keep cursor click-through-able
        gazeCursor.style.zIndex = '2147483646'; // Below heatmap canvas but above web content
        gazeCursor.style.transition = 'transform 0.1s ease-out';
        gazeCursor.style.transform = 'translate(-50%, -50%)';
        document.body.appendChild(gazeCursor);
    }

    gazeCursor.style.display = 'block';
    gazeCursor.style.left = `${px}px`;
    gazeCursor.style.top = `${py}px`;
}

// Listen for status updates
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "statusUpdate") {
        isTracking = request.isTracking;
        if (!isTracking && gazeCursor) {
            gazeCursor.style.display = 'none';
        }
    } else if (request.action === "toggleHeatmap") {
        toggleHeatmap();
    } else if (request.action === "gazeUpdate") {
        if (!isTracking) return;
        
        // Convert screen coordinates to client (viewport) & page (document) coordinates
        const cx = request.x - window.screenX;
        const cy = request.y - window.screenY;
        const px = cx + window.scrollX;
        const py = cy + window.scrollY;

        // Render cursor
        updateGazeCursor(px, py, request.showGazeCursor !== false);

        // Update tracking coordinates
        lastGazeClientX = cx;
        lastGazeClientY = cy;

        // Trace what element the gaze is currently hovering over
        const target = document.elementFromPoint(cx, cy);
        const label = getSmartLabel(target);
        handleGazeHoverChange(label);
    } else if (request.action === "heatmapTypeUpdate") {
        heatmapType = request.heatmapType;
        if (canvas) {
            // Re-render open heatmap with new filter settings
            chrome.runtime.sendMessage({ action: "getAllData" }, (response) => {
                if (response && canvas) {
                    drawHeatmapHighQuality(
                        response.heatmapData, response.trailData, response.clickData,
                        response.gazeHeatmapData, response.gazeTrailData, heatmapType
                    );
                }
            });
        }
    }
});

// Event Listeners for Mouse
function updateMousePos(e) {
    lastClientX = e.clientX;
    lastClientY = e.clientY;
    
    if (lastPageX === 0) lastPageX = e.pageX;
    if (lastPageY === 0) lastPageY = e.pageY;

    // Track the hovered element precisely
    if (isTracking) {
        const target = e.target;
        const label = getSmartLabel(target);
        handleHoverChange(label);
    }
}

document.addEventListener('mousemove', updateMousePos, true);
document.addEventListener('mouseenter', updateMousePos, true);
document.addEventListener('mouseover', updateMousePos, true);
document.addEventListener('click', updateMousePos, true);
document.addEventListener('mouseleave', () => { 
    lastClientX = 0; lastClientY = 0; 
    handleHoverChange(null);
}, true);

window.addEventListener('focus', () => {
    chrome.runtime.sendMessage({ action: "getStatus" }, (response) => {
        if (response) {
            isTracking = response.isTracking;
            showGazeCursor = response.showGazeCursor !== false;
            heatmapType = response.heatmapType || "combined";
        }
    });
});

document.addEventListener('click', (e) => {
    if (!isTracking) return;
    if (e.target === canvas) return;

    // Strict Interaction Check
    if (!isInteractable(e.target)) return;

    const x = e.pageX;
    const y = e.pageY;

    const label = getSmartLabel(e.target) || e.target.tagName;

    if (['BODY', 'HTML', 'DIV', 'SPAN'].includes(label) && !e.target.innerText.trim()) return;

    const logEntry = {
        timestamp: new Date().toLocaleTimeString(),
        x,
        y,
        gazeX: lastGazePageX > 0 ? Math.round(lastGazePageX) : null,
        gazeY: lastGazePageY > 0 ? Math.round(lastGazePageY) : null,
        element: label,
        text: e.target.innerText ? e.target.innerText.substring(0, 30).replace(/(\r\n|\n|\r)/gm, " ").trim() : '',
        url: window.location.href
    };

    chrome.runtime.sendMessage({
        action: "logData",
        click: logEntry
    });

    if (canvas) {
        drawClick(x, y);
        if (logEntry.gazeX && logEntry.gazeY && heatmapType === "combined") {
            drawGazeToClickLine(logEntry.gazeX, logEntry.gazeY, x, y);
        }
    }
}, true);

function isInteractable(el) {
    if (!el) return false;
    const tag = el.tagName.toLowerCase();

    // 1. Semantic Interactive Elements
    if (['a', 'button', 'input', 'select', 'textarea', 'details', 'summary', 'label'].includes(tag)) return true;

    // 2. Roles
    const role = el.getAttribute('role');
    if (role === 'button' || role === 'link' || role === 'menuitem' || role === 'tab') return true;

    // 3. Cursor Pointer (Computed Style)
    try {
        const style = window.getComputedStyle(el);
        if (style.cursor === 'pointer') return true;
    } catch (e) { }

    // 4. Traverse up to 3 levels
    let parent = el.parentElement;
    let depth = 0;
    while (parent && depth < 3) {
        const pTag = parent.tagName.toLowerCase();
        if (['a', 'button'].includes(pTag)) return true;
        if (parent.getAttribute('role') === 'button') return true;
        parent = parent.parentElement;
        depth++;
    }

    // 5. Code blocks
    if (tag === 'code' || tag === 'pre' || el.classList.contains('code') || el.closest('pre')) return true;

    return false;
}

function getSmartLabel(el) {
    if (!el) return null;

    const tag = el.tagName.toLowerCase();

    // 1. Ignore generic containers unless specific
    if (['body', 'html', 'main', 'div', 'span', 'section', 'article'].includes(tag)) {
        if (el.id && (el.id.includes('logo') || el.id.includes('wrapper') || el.id.includes('container'))) return null;
        if (el.className && typeof el.className === 'string' &&
            (el.className.toLowerCase().includes('logo') || el.className.toLowerCase().includes('brand'))) {
            return null;
        }

        const aria = el.getAttribute('aria-label');
        if (aria) return `Element: ${aria}`;

        // Leaf node text check
        if (el.children.length === 0) {
            const txt = el.innerText.trim();
            if (txt.length > 2 && txt.length < 50 && /[a-zA-Z0-9]/.test(txt)) {
                return `Element: ${txt}`;
            }
        }
        return null;
    }

    // 2. Interactive
    if (tag === 'a') return `Link: ${el.innerText.trim().substring(0, 30) || 'Link'}`;
    if (tag === 'button') return `Button: ${el.innerText.trim().substring(0, 30) || 'Button'}`;
    if (tag === 'input') return `Input: ${el.placeholder || el.name || el.id || 'Input'}`;
    if (tag === 'textarea') return `Input: Text Area`;
    if (tag === 'img') return `Image: ${el.alt || 'Image'}`;

    // 3. Headings / Code
    if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tag)) {
        return `${tag.toUpperCase()}: ${el.innerText.trim().substring(0, 40)}`;
    }
    if (tag === 'code' || tag === 'pre') {
        return `Code: ${el.innerText.trim().substring(0, 30)}`;
    }

    // 4. Text Content
    const text = el.innerText.trim();
    if (text && text.length > 2) {
        if (el.classList.contains('code') || el.closest('pre')) {
            return `Code: ${text.substring(0, 30)}`;
        }

        if (text.toLowerCase().includes('no message found')) return null;

        const isHex = /^#[0-9A-F]{6}$/i.test(text) || /^#[0-9A-F]{3}$/i.test(text);
        const isColorName = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'gray', 'grey', 'pink', 'brown', 'cyan', 'magenta'].includes(text.toLowerCase());
        if (isHex || isColorName) return null;

        if (text.length > 50) return `Text: ${text.substring(0, 47)}...`;
        return `Text: ${text}`;
    }

    return null;
}

async function toggleHeatmap() {
    if (canvas) {
        document.body.removeChild(canvas);
        canvas = null;
        return;
    }

    // Fetch ALL data from background
    const response = await chrome.runtime.sendMessage({ action: "getAllData" });
    if (!response) return;

    heatmapType = response.heatmapType || "combined";

    // Create canvas
    canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.zIndex = '2147483647';
    canvas.style.pointerEvents = 'none';

    canvas.width = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth, document.documentElement.offsetWidth);
    canvas.height = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight, document.documentElement.offsetHeight);

    document.body.appendChild(canvas);

    drawHeatmapHighQuality(
        response.heatmapData, response.trailData, response.clickData,
        response.gazeHeatmapData, response.gazeTrailData, heatmapType
    );
}

function drawTrail(trailData, color = 'rgba(0, 100, 255, 0.6)') {
    if (!canvas || !trailData || trailData.length < 2) return;
    const ctx = canvas.getContext('2d');

    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.0;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.moveTo(trailData[0].x, trailData[0].y);

    let i = 1;
    for (; i < trailData.length - 2; i++) {
        const xc = (trailData[i].x + trailData[i + 1].x) / 2;
        const yc = (trailData[i].y + trailData[i + 1].y) / 2;
        ctx.quadraticCurveTo(trailData[i].x, trailData[i].y, xc, yc);
    }
    if (i < trailData.length - 1) {
        ctx.quadraticCurveTo(trailData[i].x, trailData[i].y, trailData[i + 1].x, trailData[i + 1].y);
    } else if (trailData.length > 1) {
        ctx.lineTo(trailData[trailData.length - 1].x, trailData[trailData.length - 1].y);
    }

    ctx.stroke();
    ctx.restore();
}

function drawClick(x, y) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 3;
    ctx.shadowColor = 'black';
    ctx.shadowBlur = 2;

    const size = 10;
    ctx.moveTo(x - size, y - size);
    ctx.lineTo(x + size, y + size);
    ctx.moveTo(x + size, y - size);
    ctx.lineTo(x - size, y + size);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, size + 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
}

function drawGazeToClickLine(gx, gy, cx, cy) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]); // Dashed line
    ctx.moveTo(gx, gy);
    ctx.lineTo(cx, cy);
    ctx.stroke();
    
    // Draw a small red indicator at the gaze position
    ctx.beginPath();
    ctx.arc(gx, gy, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
    ctx.fill();
    ctx.restore();
}

function drawHeatmapHighQuality(heatmapData, trailData, clickData, gazeHeatmapData, gazeTrailData, type) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const currentUrl = window.location.href;

    // Filter Data by URL
    const currentTrail = trailData ? trailData.filter(p => p.url === currentUrl || !p.url) : []; 
    const currentHeatmap = heatmapData ? heatmapData.filter(p => p.url === currentUrl || !p.url) : [];

    const currentGazeTrail = gazeTrailData ? gazeTrailData.filter(p => p.url === currentUrl || !p.url) : [];
    const currentGazeHeatmap = gazeHeatmapData ? gazeHeatmapData.filter(p => p.url === currentUrl || !p.url) : [];

    // Combine all tracking points based on visualization filter type
    let allHeatPoints = [];
    
    if (type === "mouse" || type === "combined") {
        if (currentHeatmap.length > 0) allHeatPoints = allHeatPoints.concat(currentHeatmap);
        if (currentTrail.length > 0) allHeatPoints = allHeatPoints.concat(currentTrail);
    }
    
    if (type === "gaze" || type === "combined") {
        if (currentGazeHeatmap.length > 0) allHeatPoints = allHeatPoints.concat(currentGazeHeatmap);
        if (currentGazeTrail.length > 0) allHeatPoints = allHeatPoints.concat(currentGazeTrail);
    }
    
    // Weight clicks heavily too
    if (clickData && (type === "mouse" || type === "combined")) {
        const currentClicks = clickData.filter(p => !p.url || p.url === currentUrl);
        currentClicks.forEach(c => {
            for(let i=0; i<5; i++) {
                allHeatPoints.push({x: c.x, y: c.y});
            }
        });
    }

    if (allHeatPoints.length > 0) {
        // Pre-create the radial brush for performance
        const radius = 60;
        const brushCanvas = document.createElement('canvas');
        brushCanvas.width = radius * 2;
        brushCanvas.height = radius * 2;
        const brushCtx = brushCanvas.getContext('2d');
        const g = brushCtx.createRadialGradient(radius, radius, 0, radius, radius, radius);
        g.addColorStop(0, 'rgba(0, 0, 0, 0.05)'); 
        g.addColorStop(1, 'rgba(0, 0, 0, 0)');
        brushCtx.fillStyle = g;
        brushCtx.fillRect(0, 0, radius * 2, radius * 2);

        // Draw heat point for every recorded position
        allHeatPoints.forEach(point => {
            ctx.drawImage(brushCanvas, point.x - radius, point.y - radius);
        });

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        const gradientMap = createGradientMap();

        for (let i = 0; i < data.length; i += 4) {
            const alpha = data[i + 3];
            if (alpha > 0) {
                let mapIndex = Math.floor(alpha * 1.5); 
                if (mapIndex > 255) mapIndex = 255;
                
                const cIndex = mapIndex * 4;

                data[i]     = gradientMap[cIndex];     // R
                data[i + 1] = gradientMap[cIndex + 1]; // G
                data[i + 2] = gradientMap[cIndex + 2]; // B
                data[i + 3] = Math.min(255, 150 + alpha); 
            }
        }
        ctx.putImageData(imageData, 0, 0);
    }

    // Draw visual path overlays
    if (type === "mouse" || type === "combined") {
        drawTrail(currentTrail, 'rgba(30, 144, 255, 0.4)'); // Dodger Blue for mouse
    }
    if (type === "gaze" || type === "combined") {
        drawTrail(currentGazeTrail, 'rgba(255, 69, 0, 0.4)'); // Orange Red for gaze
    }

    // Draw clicks & click-gaze offsets
    if (clickData) {
        const currentClicks = clickData.filter(p => !p.url || p.url === currentUrl);
        currentClicks.forEach(c => {
            drawClick(c.x, c.y);
            if (c.gazeX && c.gazeY && type === "combined") {
                drawGazeToClickLine(c.gazeX, c.gazeY, c.x, c.y);
            }
        });
    }
}

function createGradientMap() {
    const c = document.createElement('canvas');
    c.width = 256;
    c.height = 1;
    const ctx = c.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, 256, 0);

    g.addColorStop(0.0, 'rgba(0, 0, 255, 0)');     
    g.addColorStop(0.1, 'rgba(0, 0, 255, 1)');     // Blue
    g.addColorStop(0.4, 'rgba(0, 255, 255, 1)');   // Cyan
    g.addColorStop(0.6, 'rgba(0, 255, 0, 1)');     // Green
    g.addColorStop(0.8, 'rgba(255, 255, 0, 1)');   // Yellow
    g.addColorStop(1.0, 'rgba(255, 0, 0, 1)');     // Red

    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 1);
    return ctx.getImageData(0, 0, 256, 1).data;
}

window.addEventListener('resize', () => {
    if (canvas) {
        canvas.width = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth, document.documentElement.offsetWidth);
        canvas.height = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight, document.documentElement.offsetHeight);
    }
});
