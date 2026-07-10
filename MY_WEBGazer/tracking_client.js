(function() {
  if (window.__insightux_client_loaded) return;
  window.__insightux_client_loaded = true;

  console.log('[InsightUX Guest] Client tracking script loaded.');

  let isTrackingActive = false;
  let ws = null;
  let styleTag = null;
  let sidebarContainer = null;
  let canvas = null;
  let ctx = null;

  // Offscreen heatmap caching
  let cacheCanvas = null;
  let cacheCtx = null;
  let brushCanvas = null;
  let brushCtx = null;
  let brushRadius = 55;
  let gradientCanvas = null;
  let gradCtx = null;
  let gradientMap = null;

  // Active status div
  let statusDiv = null;
  let statusTimeout = null;

  // Loop/Timer IDs
  let renderLoopId = null;
  let sampleIntervalId = null;
  let refreshIntervalId = null;
  let sidebarIntervalId = null;

  // Tracking settings
  let showGazeCursor = true;
  let showMouseCursor = true;
  let showTrails = true;
  let showHeatmap = false;
  let heatmapType = "combined";

  // Telemetry state
  let sidebarState = null;
  let lastClientX = 0, lastClientY = 0;
  let lastPageX = 0, lastPageY = 0;
  let lastSampleTime = 0;
  let stationaryStart = 0;
  let isStationary = false;
  let lastGazeSampleTime = 0;
  let lastGazePageX = 0, lastGazePageY = 0;
  let isGazeStationary = false;
  let gazeStationaryStart = 0;
  const DWELL_THRESHOLD = 3000;

  const visMousePoints = [];
  const visGazePoints = [];
  const visClicks = [];

  let trailBuffer = [];
  let heatmapBuffer = [];
  let clickBuffer = [];
  let dwellBuffer = [];
  let gazeTrailBuffer = [];
  let gazeHeatmapBuffer = [];
  let gazeDwellBuffer = [];

  let currentHoverLabel = null;
  let currentHoverStartTime = 0;
  let currentGazeHoverLabel = null;
  let currentGazeHoverStartTime = 0;

  const target = { fx: 0.5, fy: 0.5 };
  const dot = { x: null, y: null };
  const DOT_LERP = 0.07;

  // AOI state
  const aoiState = { aois: [] };
  const gazeDwellHysteresis = { pendLabel: null, pendSince: 0, activeLabel: null, emptySince: 0 };
  const DWELL_MS = 420;
  const RELEASE_MS = 1200;
  let activeBox = null;

  // Constants for element filtering
  const MIN_W = 40, MIN_H = 24, MAX_AOIS = 90, MAX_SCAN = 2500;
  const PAD_PX = 90;
  const TALL_WRAPPER_H = 220;
  const ALWAYS_SELECTOR = "nav, header, footer, img, video, iframe, h1, h2, h3, button, figure, [class*='hero'], [class*='banner'], [class*='card']";
  const TEXT_CONTAINER_SELECTOR = "div, span, section, article, main, aside, p, li";

  // Sidebar CSS Stylesheet
  const css = `
    #__insightux_sidebar_container {
      position: fixed;
      top: 0;
      right: 0;
      width: 350px;
      height: 100vh;
      z-index: 2147483647;
      display: flex;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      transform: translateX(0);
      box-sizing: border-box;
      pointer-events: auto;
    }
    #__insightux_sidebar_container.collapsed {
      transform: translateX(350px);
    }
    #__insightux_sidebar_toggle {
      width: 24px;
      height: 60px;
      background: rgba(18, 18, 24, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-right: none;
      border-radius: 8px 0 0 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      color: #ff2df0;
      font-size: 18px;
      font-weight: bold;
      align-self: center;
      box-shadow: -5px 0 15px rgba(0,0,0,0.3);
      user-select: none;
    }
    #__insightux_sidebar_panel {
      width: 350px;
      height: 100vh;
      background: rgba(18, 18, 24, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-left: 1px solid rgba(255, 255, 255, 0.15);
      box-shadow: -10px 0 30px rgba(0,0,0,0.5);
      color: #f3f4f6;
      display: flex;
      flex-direction: column;
      box-sizing: border-box;
    }
    #__insightux_sidebar_panel * {
      box-sizing: border-box;
    }
    .sidebar-header {
      padding: 20px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .sidebar-header h2 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      background: linear-gradient(135deg, #FF2DF0 0%, #7B2FBE 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.5px;
    }
    .status-indicator {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 600;
      color: rgba(255,255,255,0.6);
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background-color: #10b981;
      box-shadow: 0 0 8px #10b981;
    }
    .sidebar-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .sidebar-scroll::-webkit-scrollbar {
      width: 6px;
    }
    .sidebar-scroll::-webkit-scrollbar-track {
      background: transparent;
    }
    .sidebar-scroll::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.15);
      border-radius: 3px;
    }
    .sidebar-scroll::-webkit-scrollbar-thumb:hover {
      background: rgba(255,255,255,0.3);
    }
    .info-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .info-card {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 8px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .info-label {
      font-size: 10px;
      color: rgba(255,255,255,0.45);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 600;
    }
    .info-val {
      font-size: 16px;
      font-weight: 700;
      color: #ffffff;
    }
    #sb-timer {
      color: #ff2df0;
    }
    .section-panel {
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .section-panel h3 {
      margin: 0;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: rgba(255,255,255,0.5);
      font-weight: 700;
    }
    .control-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 500;
      color: rgba(255,255,255,0.85);
    }
    .sb-select {
      background: rgba(18, 18, 24, 0.9);
      border: 1px solid rgba(255,255,255,0.15);
      color: #fff;
      padding: 4px 8px;
      border-radius: 6px;
      outline: none;
      cursor: pointer;
      font-size: 11px;
    }
    .sb-select:hover {
      border-color: rgba(255,255,255,0.3);
    }
    .control-grid-toggles {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
    }
    .sb-toggle-btn {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 6px;
      color: rgba(255,255,255,0.6);
      padding: 8px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .sb-toggle-btn.active {
      background: rgba(123, 47, 190, 0.25);
      border-color: rgba(123, 47, 190, 0.5);
      color: #ff2df0;
      box-shadow: 0 0 10px rgba(123, 47, 190, 0.2);
    }
    .sb-toggle-btn:hover {
      background: rgba(255,255,255,0.1);
      color: #fff;
    }
    .tabs-header {
      display: flex;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      margin-bottom: 4px;
    }
    .sb-tab-btn {
      flex: 1;
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      color: rgba(255,255,255,0.45);
      padding: 8px 0;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .sb-tab-btn.active {
      color: #ff2df0;
      border-bottom-color: #ff2df0;
    }
    .sb-tab-btn:hover {
      color: #fff;
    }
    .sb-tab-content {
      display: none;
      flex-direction: column;
      gap: 10px;
    }
    .sb-tab-content.active {
      display: flex;
    }
    .sb-hero-card {
      background: linear-gradient(135deg, rgba(123, 47, 190, 0.15) 0%, rgba(255, 45, 240, 0.05) 100%);
      border: 1px solid rgba(123, 47, 190, 0.25);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      text-align: center;
    }
    .hero-lbl {
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: rgba(255,255,255,0.5);
      font-weight: 700;
    }
    .hero-el-name {
      font-size: 13px;
      font-weight: 700;
      color: #fff;
      word-break: break-all;
    }
    .hero-val {
      font-size: 20px;
      font-weight: 800;
      color: #ff2df0;
    }
    .sb-interests-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 160px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .sb-interests-list::-webkit-scrollbar {
      width: 4px;
    }
    .sb-interests-list::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.1);
      border-radius: 2px;
    }
    .sb-interest-item {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.04);
      border-radius: 6px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .sb-item-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
    }
    .sb-item-name {
      font-weight: 600;
      color: rgba(255,255,255,0.9);
      max-width: 70%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .sb-item-time {
      font-weight: 700;
      color: #ff2df0;
    }
    .sb-progress-bar-bg {
      height: 4px;
      background: rgba(255,255,255,0.06);
      border-radius: 2px;
      overflow: hidden;
    }
    .sb-progress-bar-fill {
      height: 100%;
      border-radius: 2px;
      transition: width 0.3s ease;
    }
    .gaze-fill {
      background: linear-gradient(90deg, #7B2FBE, #FF2DF0);
    }
    .mouse-fill {
      background: linear-gradient(90deg, #1E90FF, #00BFFF);
    }
    .sb-click-feed {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 200px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .sb-click-feed::-webkit-scrollbar {
      width: 4px;
    }
    .sb-click-feed::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.1);
      border-radius: 2px;
    }
    .sb-no-clicks {
      font-size: 11px;
      color: rgba(255,255,255,0.35);
      text-align: center;
      padding: 10px 0;
    }
    .sb-click-item {
      background: rgba(255,255,255,0.03);
      border-left: 3px solid #10b981;
      border-radius: 0 6px 6px 0;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 11px;
    }
    .sb-click-time-meta {
      display: flex;
      justify-content: space-between;
      color: rgba(255,255,255,0.4);
      font-size: 9px;
    }
    .sb-click-el {
      font-weight: 600;
      color: #fff;
    }
    .sb-click-text {
      color: rgba(255,255,255,0.7);
      font-style: italic;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .sb-click-coords-combo {
      display: flex;
      flex-direction: column;
      gap: 2px;
      margin-top: 2px;
      padding-top: 4px;
      border-top: 1px solid rgba(255,255,255,0.05);
      font-size: 9px;
    }
    .sb-coord-line {
      display: flex;
      justify-content: space-between;
    }
    .sb-coord-lbl {
      color: rgba(255,255,255,0.4);
    }
    .sb-coord-val {
      font-weight: 600;
    }
    .sb-mouse-coord {
      color: #00BFFF;
    }
    .sb-gaze-coord {
      color: #ff2df0;
    }
    .sb-offset-val {
      color: #10b981;
      font-weight: 700;
    }
  `;

  // ─── Telemetry & DOM Parsing Helpers ───
  function isInteractable(el) {
    if (!el) return false;
    if (el.closest && el.closest('#__insightux_sidebar_container')) return false;
    const tag = el.tagName.toLowerCase();
    if (['a', 'button', 'input', 'select', 'textarea', 'details', 'summary', 'label'].includes(tag)) return true;
    const role = el.getAttribute('role');
    if (role === 'button' || role === 'link' || role === 'menuitem' || role === 'tab') return true;
    try {
      const style = window.getComputedStyle(el);
      if (style.cursor === 'pointer') return true;
    } catch (e) { }
    let parent = el.parentElement;
    let depth = 0;
    while (parent && depth < 3) {
      const pTag = parent.tagName.toLowerCase();
      if (['a', 'button'].includes(pTag)) return true;
      if (parent.getAttribute('role') === 'button') return true;
      parent = parent.parentElement;
      depth++;
    }
    if (tag === 'code' || tag === 'pre' || el.classList.contains('code') || el.closest('pre')) return true;
    return false;
  }

  function getSmartLabel(el) {
    if (!el) return null;
    if (el.closest && el.closest('#__insightux_sidebar_container')) return null;
    const tag = el.tagName.toLowerCase();
    if (['body', 'html', 'main', 'div', 'span', 'section', 'article'].includes(tag)) {
      if (el.id && (el.id.includes('logo') || el.id.includes('wrapper') || el.id.includes('container'))) return null;
      if (el.className && typeof el.className === 'string' &&
          (el.className.toLowerCase().includes('logo') || el.className.toLowerCase().includes('brand'))) {
        return null;
      }
      const aria = el.getAttribute('aria-label');
      if (aria) return 'Element: ' + aria;
      if (el.children.length === 0) {
        const txt = el.innerText.trim();
        if (txt.length > 2 && txt.length < 50 && /[a-zA-Z0-9]/.test(txt)) {
          return 'Element: ' + txt;
        }
      }
      return null;
    }
    if (tag === 'a') return 'Link: ' + (el.innerText.trim().substring(0, 30) || 'Link');
    if (tag === 'button') return 'Button: ' + (el.innerText.trim().substring(0, 30) || 'Button');
    if (tag === 'input') return 'Input: ' + (el.placeholder || el.name || el.id || 'Input');
    if (tag === 'textarea') return 'Input: Text Area';
    if (tag === 'img') return 'Image: ' + (el.alt || 'Image');
    if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tag)) {
      return tag.toUpperCase() + ': ' + el.innerText.trim().substring(0, 40);
    }
    if (tag === 'code' || tag === 'pre') {
      return 'Code: ' + el.innerText.trim().substring(0, 30);
    }
    const text = el.innerText.trim();
    if (text && text.length > 2) {
      if (el.classList.contains('code') || el.closest('pre')) {
        return 'Code: ' + text.substring(0, 30);
      }
      if (text.toLowerCase().includes('no message found')) return null;
      const isHex = /^#[0-9A-F]{6}$/i.test(text) || /^#[0-9A-F]{3}$/i.test(text);
      const isColorName = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'gray', 'grey', 'pink', 'brown', 'cyan', 'magenta'].includes(text.toLowerCase());
      if (isHex || isColorName) return null;
      if (text.length > 50) return 'Text: ' + text.substring(0, 47) + '...';
      return 'Text: ' + text;
    }
    return null;
  }

  function hasOwnText(el){
    for (const node of el.childNodes){
      if (node.nodeType === 3 && node.textContent.trim().length > 2) return true;
    }
    return false;
  }

  function labelFor(el){
    if (el.dataset && el.dataset.aoi) return el.dataset.aoi.slice(0,40);
    const tag = el.tagName.toLowerCase();
    if (tag==='nav') return 'navbar';
    if (tag==='header') return 'header';
    if (tag==='footer') return 'footer';
    if (tag==='img'){
      const alt=(el.getAttribute('alt')||'').trim();
      if (alt) return 'img: '+alt.slice(0,30);
      const src=el.getAttribute('src')||'';
      const name=src.split('/').pop().split('?')[0];
      return 'img: '+(name||'image').slice(0,30);
    }
    if (tag==='video') return 'video';
    if (tag==='iframe') return 'embed';
    if (tag==='h1'||tag==='h2'){
      const t=(el.innerText||'').trim().replace(/\s+/g,' ');
      if (t) return tag+': '+t.slice(0,30);
    }
    const id=el.id?('#'+el.id):'';
    let cls='';
    if (el.className && typeof el.className==='string'){
      const f=el.className.trim().split(/\s+/)[0];
      if (f) cls='.'+f;
    }
    const txt=(el.innerText||'').trim().replace(/\s+/g,' ').slice(0,24);
    const base=id||cls||tag;
    return txt?(base+' ('+txt+')'):base;
  }

  function refreshAOIs() {
    const vh = window.innerHeight;
    const raw = [];
    const seen = new Set();
    let full = false;

    function consider(el){
      if (full) return;
      if (el.closest && el.closest('#__insightux_sidebar_container')) return;
      const tag = el.tagName.toLowerCase();
      const explicit = !!(el.dataset && el.dataset.aoi);
      if (!explicit){
        const isAlways = (tag==='nav'||tag==='header'||tag==='footer'||tag==='img'||
                           tag==='video'||tag==='iframe'||
                           tag==='h1'||tag==='h2'||tag==='h3'||tag==='button'||tag==='figure') ||
                          (el.className && typeof el.className==='string' &&
                           /hero|banner|card/i.test(el.className));
        if (!isAlways && !hasOwnText(el)) return;
      }
      const r = el.getBoundingClientRect();
      if (r.width<MIN_W || r.height<MIN_H) return;
      if (r.bottom<0 || r.top>vh) return;
      const label = labelFor(el);
      if (!label) return;
      const key = label+'@'+Math.round(r.left)+','+Math.round(r.top);
      if (seen.has(key)) return;
      seen.add(key);
      const pos = getComputedStyle(el).position;
      raw.push({el:el, label:label, x:Math.round(r.left), y:Math.round(r.top),
                w:Math.round(r.width), h:Math.round(r.height),
                sticky:(pos==='sticky'||pos==='fixed')});
      if (raw.length >= MAX_AOIS*2) full = true;
    }

    document.querySelectorAll('[data-aoi]').forEach(consider);
    document.querySelectorAll(ALWAYS_SELECTOR).forEach(consider);

    const candidates = document.querySelectorAll(TEXT_CONTAINER_SELECTOR);
    for (let i=0; i<candidates.length && i<MAX_SCAN && !full; i++){
      consider(candidates[i]);
    }

    const filtered = raw.filter(o => {
      const isTallWrapper = o.h > TALL_WRAPPER_H &&
                             raw.some(o2 => o2 !== o && o.el.contains(o2.el));
      return !isTallWrapper;
    });

    aoiState.aois = filtered.slice(0, MAX_AOIS).map(o => ({
      label:o.label, x:o.x, y:o.y, w:o.w, h:o.h, sticky:o.sticky
    }));
  }

  // ─── Interaction logs ───
  function handleHoverChange(newLabel) {
    if (!isTrackingActive) return;
    if (newLabel !== currentHoverLabel) {
      const now = Date.now();
      if (currentHoverLabel && currentHoverStartTime > 0) {
        const duration = now - currentHoverStartTime;
        if (duration > 10) {
          dwellBuffer.push({ element: currentHoverLabel, duration: duration });
          sidebarState.mouseDwells[currentHoverLabel] = (sidebarState.mouseDwells[currentHoverLabel] || 0) + duration;
        }
      }
      currentHoverLabel = newLabel;
      currentHoverStartTime = newLabel ? now : 0;
    }
  }

  function handleGazeHoverChange(newLabel) {
    if (!isTrackingActive) return;
    if (newLabel !== currentGazeHoverLabel) {
      const now = Date.now();
      if (currentGazeHoverLabel && currentGazeHoverStartTime > 0) {
        const duration = now - currentGazeHoverStartTime;
        if (duration > 10) {
          gazeDwellBuffer.push({ element: currentGazeHoverLabel, duration: duration });
          sidebarState.gazeDwells[currentGazeHoverLabel] = (sidebarState.gazeDwells[currentGazeHoverLabel] || 0) + duration;
        }
      }
      currentGazeHoverLabel = newLabel;
      currentGazeHoverStartTime = newLabel ? now : 0;
    }
  }

  function updateMousePos(e) {
    if (e.target && e.target.closest && e.target.closest('#__insightux_sidebar_container')) {
      lastClientX = 0; lastClientY = 0;
      handleHoverChange(null);
      return;
    }
    lastClientX = e.clientX;
    lastClientY = e.clientY;
    lastPageX = e.pageX;
    lastPageY = e.pageY;
    
    const label = getSmartLabel(e.target);
    handleHoverChange(label);
  }

  function handlePageClick(e) {
    if (e.target === canvas) return;
    if (e.target.closest && e.target.closest('#__insightux_sidebar_container')) return;
    if (!isInteractable(e.target)) return;

    const label = getSmartLabel(e.target) || e.target.tagName;
    if (['BODY', 'HTML', 'DIV', 'SPAN'].includes(label) && !e.target.innerText.trim()) return;

    const logEntry = {
      timestamp: new Date().toLocaleTimeString(),
      x: e.pageX,
      y: e.pageY,
      gazeX: lastGazePageX > 0 ? Math.round(lastGazePageX) : null,
      gazeY: lastGazePageY > 0 ? Math.round(lastGazePageY) : null,
      element: label,
      text: e.target.innerText ? e.target.innerText.substring(0, 30).replace(/(\r\n|\n|\r)/gm, " ").trim() : '',
      url: window.location.href
    };

    clickBuffer.push(logEntry);
    visClicks.push(logEntry);
    if (visClicks.length > 100) visClicks.shift();

    sidebarState.clicksCount++;
    sidebarState.clicksList.unshift(logEntry);
    if (sidebarState.clicksList.length > 30) sidebarState.clicksList.pop();

    if (showHeatmap) {
      updateHeatmapCache();
    }
  }

  // ─── Heatmap Caches ───
  function resize() {
    if (canvas) {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
  }

  function resizeCache() {
    if (cacheCanvas) {
      cacheCanvas.width = window.innerWidth;
      cacheCanvas.height = window.innerHeight;
    }
  }

  function updateHeatmapCache() {
    if (!cacheCtx || !cacheCanvas) return;
    cacheCtx.clearRect(0, 0, cacheCanvas.width, cacheCanvas.height);
    
    let points = [];
    const scrollX = window.scrollX;
    const scrollY = window.scrollY;

    if (heatmapType === "mouse" || heatmapType === "combined") {
      points = points.concat(visMousePoints);
    }
    if (heatmapType === "gaze" || heatmapType === "combined") {
      points = points.concat(visGazePoints);
    }

    if (points.length === 0) return;

    points.forEach(p => {
      const vx = p.x - scrollX;
      const vy = p.y - scrollY;
      if (vx >= -brushRadius && vx <= cacheCanvas.width + brushRadius &&
          vy >= -brushRadius && vy <= cacheCanvas.height + brushRadius) {
        cacheCtx.drawImage(brushCanvas, vx - brushRadius, vy - brushRadius);
      }
    });

    try {
      const imgData = cacheCtx.getImageData(0, 0, cacheCanvas.width, cacheCanvas.height);
      const pix = imgData.data;
      for (let i = 0; i < pix.length; i += 4) {
        const a = pix[i + 3];
        if (a > 0) {
          let idx = Math.floor(a * 1.6);
          if (idx > 255) idx = 255;
          const cOffset = idx * 4;
          pix[i]     = gradientMap[cOffset];
          pix[i + 1] = gradientMap[cOffset + 1];
          pix[i + 2] = gradientMap[cOffset + 2];
          pix[i + 3] = Math.min(255, 140 + a);
        }
      }
      cacheCtx.putImageData(imgData, 0, 0);
    } catch(e) {
      console.error("Heatmap rendering failed:", e);
    }
  }

  // ─── Live Render Canvas Loop ───

  function renderLoop() {
    if (!isTrackingActive || !ctx || !canvas) return;

    const w = window.innerWidth, h = window.innerHeight;
    const tx = target.fx * w, ty = target.fy * h;
    if (dot.x === null) { dot.x = tx; dot.y = ty; }
    dot.x += (tx - dot.x) * DOT_LERP;
    dot.y += (ty - dot.y) * DOT_LERP;

    const now = performance.now();
    const px = dot.x, py = dot.y;
    const scrollX = window.scrollX;
    const scrollY = window.scrollY;

    // Gaze box candidate
    let cand = null;
    for (const a of aoiState.aois) {
      if (px >= a.x-PAD_PX && px <= a.x+a.w+PAD_PX && py >= a.y-PAD_PX && py <= a.y+a.h+PAD_PX) {
        if (!cand || (a.w * a.h) < (cand.w * cand.h)) cand = a;
      }
    }

    // Dwell + hysteresis
    const candLabel = cand ? cand.label : null;
    if (candLabel !== gazeDwellHysteresis.pendLabel) {
      gazeDwellHysteresis.pendLabel = candLabel;
      gazeDwellHysteresis.pendSince = now;
    }
    if (cand) {
      gazeDwellHysteresis.emptySince = 0;
      if (gazeDwellHysteresis.activeLabel !== cand.label && (now - gazeDwellHysteresis.pendSince) >= DWELL_MS) {
        gazeDwellHysteresis.activeLabel = cand.label;
      }
    } else {
      if (gazeDwellHysteresis.emptySince === 0) gazeDwellHysteresis.emptySince = now;
      if (gazeDwellHysteresis.activeLabel && (now - gazeDwellHysteresis.emptySince) >= RELEASE_MS) {
        gazeDwellHysteresis.activeLabel = null;
      }
    }

    let active = null;
    if (gazeDwellHysteresis.activeLabel) {
      for (const a of aoiState.aois) {
        if (a.label === gazeDwellHysteresis.activeLabel) {
          active = a;
          break;
        }
      }
      if (!active) gazeDwellHysteresis.activeLabel = null;
    }
    activeBox = active;

    // Update Gaze Hover Dwell State
    handleGazeHoverChange(activeBox ? activeBox.label : null);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Heatmap (if enabled)
    if (showHeatmap) {
      ctx.drawImage(cacheCanvas, 0, 0);
    }

    // 2. Draw Clicks & lines
    visClicks.forEach(c => {
      const cx = c.x - scrollX;
      const cy = c.y - scrollY;
      
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.arc(cx, cy, 14, 0, 2*Math.PI);
      ctx.moveTo(cx - 9, cy - 9);
      ctx.lineTo(cx + 9, cy + 9);
      ctx.moveTo(cx + 9, cy - 9);
      ctx.lineTo(cx - 9, cy + 9);
      ctx.stroke();

      if (c.gazeX && c.gazeY && heatmapType === "combined") {
        const gcx = c.gazeX - scrollX;
        const gcy = c.gazeY - scrollY;
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.55)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 3]);
        ctx.moveTo(gcx, gcy);
        ctx.lineTo(cx, cy);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(gcx, gcy, 4, 0, 2*Math.PI);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
        ctx.fill();
      }
      ctx.restore();
    });

    // 3. Draw Trails
    if (showTrails) {
      if (showMouseCursor && visMousePoints.length > 1) {
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(30, 144, 255, 0.55)';
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        let moved = false;
        visMousePoints.forEach((p, idx) => {
          const vx = p.x - scrollX;
          const vy = p.y - scrollY;
          if (idx === 0) ctx.moveTo(vx, vy);
          else ctx.lineTo(vx, vy);
          moved = true;
        });
        if (moved) ctx.stroke();
        ctx.restore();
      }
      if (showGazeCursor && visGazePoints.length > 1) {
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 69, 0, 0.45)';
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        let moved = false;
        visGazePoints.forEach((p, idx) => {
          const vx = p.x - scrollX;
          const vy = p.y - scrollY;
          if (idx === 0) ctx.moveTo(vx, vy);
          else ctx.lineTo(vx, vy);
          moved = true;
        });
        if (moved) ctx.stroke();
        ctx.restore();
      }
    }

    // 4. Draw Active Gaze Highlight (AOI)
    if (activeBox) {
      ctx.fillStyle = 'rgba(123,47,190,0.16)';
      ctx.fillRect(activeBox.x, activeBox.y, activeBox.w, activeBox.h);
      ctx.strokeStyle = '#7B2FBE'; ctx.lineWidth = 3;
      ctx.strokeRect(activeBox.x, activeBox.y, activeBox.w, activeBox.h);

      const label = activeBox.label;
      ctx.font = 'bold 13px Arial';
      const tw = ctx.measureText(label).width + 16;
      const ly = Math.max(0, activeBox.y - 22);
      ctx.fillStyle = '#7B2FBE';
      ctx.fillRect(activeBox.x, ly, tw, 20);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, activeBox.x + 8, ly + 14);
    }

    // 5. Draw Gaze Cursor Dot
    if (showGazeCursor) {
      ctx.beginPath();
      ctx.arc(dot.x, dot.y, 13, 0, 2*Math.PI);
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(dot.x, dot.y, 9, 0, 2*Math.PI);
      ctx.fillStyle = '#FF2DF0';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(0,0,0,0.55)';
      ctx.stroke();
    }

    // 6. Draw Mouse Cursor Dot
    if (showMouseCursor && lastClientX !== 0) {
      ctx.beginPath();
      ctx.arc(lastClientX, lastClientY, 8, 0, 2*Math.PI);
      ctx.fillStyle = 'rgba(30, 144, 255, 0.8)';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.stroke();
    }

    renderLoopId = requestAnimationFrame(renderLoop);
  }

  // ─── Keybind listeners for controls ───
  function handleKeyDown(e) {
    if (e.key.toLowerCase() === 'h') {
      showHeatmap = !showHeatmap;
      if (showHeatmap) updateHeatmapCache();
      showStatusIndicator("Heatmap: " + (showHeatmap ? "ON" : "OFF"));
      const btn = document.getElementById('sb-toggle-heatmap');
      if (btn) btn.classList.toggle('active', showHeatmap);
    }
    if (e.key.toLowerCase() === 't') {
      showTrails = !showTrails;
      showStatusIndicator("Trails: " + (showTrails ? "ON" : "OFF"));
      const btn = document.getElementById('sb-toggle-trails');
      if (btn) btn.classList.toggle('active', showTrails);
    }
    if (e.key.toLowerCase() === 'c') {
      showGazeCursor = !showGazeCursor;
      showStatusIndicator("Gaze Cursor: " + (showGazeCursor ? "ON" : "OFF"));
      const btn = document.getElementById('sb-toggle-gaze-cursor');
      if (btn) btn.classList.toggle('active', showGazeCursor);
    }
    if (e.key.toLowerCase() === 'm') {
      showMouseCursor = !showMouseCursor;
      showStatusIndicator("Mouse Tracker: " + (showMouseCursor ? "ON" : "OFF"));
      const btn = document.getElementById('sb-toggle-mouse-cursor');
      if (btn) btn.classList.toggle('active', showMouseCursor);
    }
    if (e.key.toLowerCase() === 'y') {
      if (heatmapType === "combined") heatmapType = "mouse";
      else if (heatmapType === "mouse") heatmapType = "gaze";
      else heatmapType = "combined";
      if (showHeatmap) updateHeatmapCache();
      showStatusIndicator("Heatmap Mode: " + heatmapType.toUpperCase());
      const sel = document.getElementById('sb-heatmap-type');
      if (sel) sel.value = heatmapType;
    }
  }

  function showStatusIndicator(msg) {
    if (!statusDiv) return;
    statusDiv.textContent = msg;
    statusDiv.style.display = 'block';
    if (statusTimeout) clearTimeout(statusTimeout);
    statusTimeout = setTimeout(() => {
      statusDiv.style.display = 'none';
    }, 1500);
  }

  // ─── WebSocket Client for Python Gaze Coordinates ───
  function connectGazeWS() {
    if (!isTrackingActive) return;
    
    console.log('[InsightUX Guest] Connecting to Gaze Python WS server...');
    ws = new WebSocket("ws://127.0.0.1:8765");

    ws.onopen = () => {
      console.log('[InsightUX Guest] Connected to Gaze Python server.');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "gaze") {
          // Normalize coordinates (sx, sy are screen absolute pixels)
          // Primary screen size is screen.width / screen.height
          const sw = window.screen.width;
          const sh = window.screen.height;
          
          const fx = data.x / sw;
          const fy = data.y / sh;
          
          target.fx = fx;
          target.fy = fy;
        }
      } catch(e) {
        console.error('[InsightUX Guest] Error parsing gaze WS message:', e);
      }
    };

    ws.onclose = () => {
      console.log('[InsightUX Guest] Gaze WS closed.');
      if (isTrackingActive) {
        setTimeout(connectGazeWS, 1500); // Auto reconnect in 1.5s
      }
    };

    ws.onerror = (err) => {
      console.error('[InsightUX Guest] Gaze WS error:', err);
    };
  }

  // ─── Sidebar Updates ───
  let isSidebarCollapsed = false;

  function initSidebarEvents() {
    const container = document.getElementById('__insightux_sidebar_container');
    const toggle = document.getElementById('__insightux_sidebar_toggle');
    const toggleArrow = toggle ? toggle.querySelector('span') : null;

    if (toggle && container) {
      toggle.addEventListener('click', () => {
        isSidebarCollapsed = !isSidebarCollapsed;
        container.classList.toggle('collapsed', isSidebarCollapsed);
        if (toggleArrow) {
          toggleArrow.textContent = isSidebarCollapsed ? '‹' : '›';
        }
      });
    }

    const tabGaze = document.getElementById('sb-tab-gaze');
    const tabMouse = document.getElementById('sb-tab-mouse');
    const contentGaze = document.getElementById('sb-content-gaze');
    const contentMouse = document.getElementById('sb-content-mouse');

    if (tabGaze && tabMouse && contentGaze && contentMouse) {
      tabGaze.addEventListener('click', () => {
        tabGaze.classList.add('active');
        tabMouse.classList.remove('active');
        contentGaze.classList.add('active');
        contentMouse.classList.remove('active');
      });
      tabMouse.addEventListener('click', () => {
        tabMouse.classList.add('active');
        tabGaze.classList.remove('active');
        contentMouse.classList.add('active');
        contentGaze.classList.remove('active');
      });
    }

    const selectHeatmapType = document.getElementById('sb-heatmap-type');
    const btnHeatmap = document.getElementById('sb-toggle-heatmap');
    const btnTrails = document.getElementById('sb-toggle-trails');
    const btnGaze = document.getElementById('sb-toggle-gaze-cursor');
    const btnMouse = document.getElementById('sb-toggle-mouse-cursor');

    if (selectHeatmapType) {
      selectHeatmapType.value = heatmapType;
      selectHeatmapType.addEventListener('change', (e) => {
        heatmapType = e.target.value;
        if (showHeatmap) updateHeatmapCache();
      });
    }

    if (btnHeatmap) {
      btnHeatmap.classList.toggle('active', showHeatmap);
      btnHeatmap.addEventListener('click', () => {
        showHeatmap = !showHeatmap;
        btnHeatmap.classList.toggle('active', showHeatmap);
        if (showHeatmap) updateHeatmapCache();
      });
    }

    if (btnTrails) {
      btnTrails.classList.toggle('active', showTrails);
      btnTrails.addEventListener('click', () => {
        showTrails = !showTrails;
        btnTrails.classList.toggle('active', showTrails);
      });
    }

    if (btnGaze) {
      btnGaze.classList.toggle('active', showGazeCursor);
      btnGaze.addEventListener('click', () => {
        showGazeCursor = !showGazeCursor;
        btnGaze.classList.toggle('active', showGazeCursor);
      });
    }

    if (btnMouse) {
      btnMouse.classList.toggle('active', showMouseCursor);
      btnMouse.addEventListener('click', () => {
        showMouseCursor = !showMouseCursor;
        btnMouse.classList.toggle('active', showMouseCursor);
      });
    }
  }

  function renderDwells(type) {
    const dwells = type === 'gaze' ? sidebarState.gazeDwells : sidebarState.mouseDwells;
    const heroEl = document.getElementById(`sb-${type}-hero`);
    const heroElName = document.getElementById(`sb-${type}-hero-el`);
    const heroVal = document.getElementById(`sb-${type}-hero-val`);
    const listContainer = document.getElementById(`sb-${type}-list`);

    if (!listContainer) return;

    const sorted = Object.entries(dwells)
      .map(([name, duration]) => ({ name, duration }))
      .filter(item => item.duration > 0)
      .sort((a, b) => b.duration - a.duration);

    if (sorted.length === 0) {
      if (heroEl) heroEl.style.display = 'none';
      listContainer.innerHTML = '<div style="font-size:11px;color:rgba(255,255,255,0.35);text-align:center;padding:10px 0;">No dwells recorded yet.</div>';
      return;
    }

    const topItem = sorted[0];
    if (heroEl && heroElName && heroVal) {
      heroEl.style.display = 'flex';
      heroElName.textContent = topItem.name;
      heroVal.textContent = (topItem.duration / 1000).toFixed(1) + 's';
    }

    const maxDuration = topItem.duration;
    let listHTML = '';
    sorted.slice(0, 5).forEach(item => {
      const pct = maxDuration > 0 ? (item.duration / maxDuration) * 100 : 0;
      const fillClass = type === 'gaze' ? 'gaze-fill' : 'mouse-fill';
      listHTML += `
        <div class="sb-interest-item">
          <div class="sb-item-meta">
            <span class="sb-item-name" title="${item.name}">${item.name}</span>
            <span class="sb-item-time">${(item.duration / 1000).toFixed(1)}s</span>
          </div>
          <div class="sb-progress-bar-bg">
            <div class="sb-progress-bar-fill ${fillClass}" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    });
    listContainer.innerHTML = listHTML;
  }

  function renderClicksFeed() {
    const container = document.getElementById('sb-click-feed-container');
    if (!container) return;

    if (sidebarState.clicksList.length === 0) {
      container.innerHTML = '<div class="sb-no-clicks">No clicks recorded yet.</div>';
      return;
    }

    let html = '';
    sidebarState.clicksList.slice(0, 15).forEach(click => {
      let offsetHTML = '';
      if (click.gazeX !== null && click.gazeY !== null) {
        const dx = click.x - click.gazeX;
        const dy = click.y - click.gazeY;
        const dist = Math.round(Math.sqrt(dx*dx + dy*dy));
        offsetHTML = `
          <div class="sb-coord-line">
            <span class="sb-coord-lbl">Gaze Coords:</span>
            <span class="sb-coord-val sb-gaze-coord">(${click.gazeX}, ${click.gazeY})</span>
          </div>
          <div class="sb-coord-line">
            <span class="sb-coord-lbl">Eye-Mouse Offset:</span>
            <span class="sb-offset-val">${dist}px</span>
          </div>
        `;
      }
      
      html += `
        <div class="sb-click-item">
          <div class="sb-click-time-meta">
            <span>Click Event</span>
            <span>${click.timestamp}</span>
          </div>
          <span class="sb-click-el">${click.element}</span>
          ${click.text ? `<span class="sb-click-text">"${click.text}"</span>` : ''}
          <div class="sb-click-coords-combo">
            <div class="sb-coord-line">
              <span class="sb-coord-lbl">Mouse Coords:</span>
              <span class="sb-coord-val sb-mouse-coord">(${click.x}, ${click.y})</span>
            </div>
            ${offsetHTML}
          </div>
        </div>
      `;
    });
    container.innerHTML = html;
  }

  function updateSidebarUI() {
    if (!sidebarState) return;
    const elapsedSec = Math.floor((Date.now() - sidebarState.startTime) / 1000);
    const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
    const secs = String(elapsedSec % 60).padStart(2, '0');
    const timerEl = document.getElementById('sb-timer');
    if (timerEl) timerEl.textContent = `${mins}:${secs}`;

    const gazeCountEl = document.getElementById('sb-gaze-count');
    if (gazeCountEl) gazeCountEl.textContent = sidebarState.gazeDataCount;
    const mouseCountEl = document.getElementById('sb-mouse-count');
    if (mouseCountEl) mouseCountEl.textContent = sidebarState.mouseDataCount;
    const clickCountEl = document.getElementById('sb-click-count');
    if (clickCountEl) clickCountEl.textContent = sidebarState.clicksCount;

    renderDwells('gaze');
    renderDwells('mouse');
    renderClicksFeed();
  }

  // ─── Tracking Lifecycles ───
  function startInsightUXTracking() {
    if (isTrackingActive) return;
    isTrackingActive = true;

    console.log('[InsightUX Guest] Initiating eye and mouse tracking session...');

    sidebarState = {
      startTime: Date.now(),
      mouseDataCount: 0,
      gazeDataCount: 0,
      clicksCount: 0,
      clicksList: [],
      gazeDwells: {},
      mouseDwells: {}
    };

    // Inject Stylesheet
    styleTag = document.createElement('style');
    styleTag.textContent = css;
    document.head.appendChild(styleTag);

    // Create Canvas Overlay
    canvas = document.createElement('canvas');
    canvas.id = '__insightux_canvas';
    canvas.style.cssText = 'position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483646;';
    (document.body || document.documentElement).appendChild(canvas);
    ctx = canvas.getContext('2d');
    resize();
    window.addEventListener('resize', resize, { passive: true });

    // Create Heatmap Buffers
    cacheCanvas = document.createElement('canvas');
    cacheCtx = cacheCanvas.getContext('2d');
    resizeCache();
    window.addEventListener('resize', resizeCache, { passive: true });

    brushRadius = 55;
    brushCanvas = document.createElement('canvas');
    brushCanvas.width = brushRadius * 2;
    brushCanvas.height = brushRadius * 2;
    brushCtx = brushCanvas.getContext('2d');
    const brushGrad = brushCtx.createRadialGradient(brushRadius, brushRadius, 0, brushRadius, brushRadius, brushRadius);
    brushGrad.addColorStop(0, 'rgba(0,0,0,0.06)');
    brushGrad.addColorStop(1, 'rgba(0,0,0,0)');
    brushCtx.fillStyle = brushGrad;
    brushCtx.fillRect(0, 0, brushRadius * 2, brushRadius * 2);

    gradientCanvas = document.createElement('canvas');
    gradientCanvas.width = 256;
    gradientCanvas.height = 1;
    gradCtx = gradientCanvas.getContext('2d');
    const grad = gradCtx.createLinearGradient(0, 0, 256, 0);
    grad.addColorStop(0.0, 'rgba(0, 0, 255, 0)');
    grad.addColorStop(0.1, 'rgba(0, 0, 255, 1)');     // Blue
    grad.addColorStop(0.4, 'rgba(0, 255, 255, 1)');   // Cyan
    grad.addColorStop(0.6, 'rgba(0, 255, 0, 1)');     // Green
    grad.addColorStop(0.8, 'rgba(255, 255, 0, 1)');   // Yellow
    grad.addColorStop(1.0, 'rgba(255, 0, 0, 1)');     // Red
    gradCtx.fillStyle = grad;
    gradCtx.fillRect(0, 0, 256, 1);
    gradientMap = gradCtx.getImageData(0, 0, 256, 1).data;

    // Create Sidebar Panel
    sidebarContainer = document.createElement('div');
    sidebarContainer.id = '__insightux_sidebar_container';
    sidebarContainer.innerHTML = `
      <div id="__insightux_sidebar_toggle"><span>›</span></div>
      <div id="__insightux_sidebar_panel">
        <div class="sidebar-header">
          <h2>InsightUX Analytics</h2>
          <div class="status-indicator">
            <span class="status-dot"></span>
            <span>Gaze Server Connected</span>
          </div>
        </div>
        <div class="sidebar-scroll">
          <div class="info-grid">
            <div class="info-card">
              <span class="info-label">Session Time</span>
              <span class="info-val" id="sb-timer">00:00</span>
            </div>
            <div class="info-card">
              <span class="info-label">Gaze Points</span>
              <span class="info-val" id="sb-gaze-count">0</span>
            </div>
            <div class="info-card">
              <span class="info-label">Mouse Points</span>
              <span class="info-val" id="sb-mouse-count">0</span>
            </div>
            <div class="info-card">
              <span class="info-label">Clicks</span>
              <span class="info-val" id="sb-click-count">0</span>
            </div>
          </div>
          <div class="section-panel">
            <h3>Visualization Settings</h3>
            <div class="control-row">
              <label>Heatmap Mode</label>
              <select id="sb-heatmap-type" class="sb-select">
                <option value="combined">Combined</option>
                <option value="gaze">Gaze Only</option>
                <option value="mouse">Mouse Only</option>
              </select>
            </div>
            <div class="control-grid-toggles">
              <button class="sb-toggle-btn" id="sb-toggle-heatmap">Heatmap</button>
              <button class="sb-toggle-btn active" id="sb-toggle-trails">Trails</button>
              <button class="sb-toggle-btn active" id="sb-toggle-gaze-cursor">Gaze Dot</button>
              <button class="sb-toggle-btn active" id="sb-toggle-mouse-cursor">Mouse Dot</button>
            </div>
          </div>
          <div class="section-panel">
            <div class="tabs-header">
              <button id="sb-tab-gaze" class="sb-tab-btn active">Gaze Dwell</button>
              <button id="sb-tab-mouse" class="sb-tab-btn">Mouse Hover</button>
            </div>
            <div id="sb-content-gaze" class="sb-tab-content active">
              <div class="sb-hero-card" id="sb-gaze-hero" style="display:none;">
                <span class="hero-lbl">Most Looked Element</span>
                <span class="hero-el-name" id="sb-gaze-hero-el">-</span>
                <span class="hero-val" id="sb-gaze-hero-val">0.0s</span>
              </div>
              <div class="sb-interests-list" id="sb-gaze-list"></div>
            </div>
            <div id="sb-content-mouse" class="sb-tab-content">
              <div class="sb-hero-card" id="sb-mouse-hero" style="display:none;">
                <span class="hero-lbl">Most Hovered Element</span>
                <span class="hero-el-name" id="sb-mouse-hero-el">-</span>
                <span class="hero-val" id="sb-mouse-hero-val">0.0s</span>
              </div>
              <div class="sb-interests-list" id="sb-mouse-list"></div>
            </div>
          </div>
          <div class="section-panel">
            <h3>Recent Clicks Log</h3>
            <div class="sb-click-feed" id="sb-click-feed-container">
              <div class="sb-no-clicks">No clicks recorded yet.</div>
            </div>
          </div>
        </div>
      </div>
    `;
    (document.body || document.documentElement).appendChild(sidebarContainer);

    // Create Status Indicators
    statusDiv = document.createElement('div');
    statusDiv.style.cssText = 'position:fixed;top:20px;right:20px;background:rgba(0,0,0,0.85);color:#fff;padding:8px 16px;border-radius:4px;font-family:Arial,sans-serif;font-size:14px;z-index:2147483647;pointer-events:none;display:none;border:1px solid #7B2FBE;';
    document.body.appendChild(statusDiv);

    // Initialize Event Listeners
    initSidebarEvents();
    document.addEventListener('mousemove', updateMousePos, true);
    document.addEventListener('mouseenter', updateMousePos, true);
    document.addEventListener('mouseover', updateMousePos, true);
    document.addEventListener('click', handlePageClick, true);
    document.addEventListener('keydown', handleKeyDown, true);

    window.insightuxAOIs = function() {
      return JSON.stringify({
        url: location.href,
        scrollX: Math.round(window.scrollX),
        scrollY: Math.round(window.scrollY),
        viewport: { w: window.innerWidth, h: window.innerHeight },
        page: { w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight },
        aois: aoiState.aois
      });
    };

    window.insightuxMouseData = function() {
      const now = Date.now();
      if (currentHoverLabel && currentHoverStartTime > 0) {
        const duration = now - currentHoverStartTime;
        if (duration > 50) {
          dwellBuffer.push({ element: currentHoverLabel, duration: duration });
          sidebarState.mouseDwells[currentHoverLabel] = (sidebarState.mouseDwells[currentHoverLabel] || 0) + duration;
          currentHoverStartTime = now;
        }
      }
      if (currentGazeHoverLabel && currentGazeHoverStartTime > 0) {
        const durationGaze = now - currentGazeHoverStartTime;
        if (durationGaze > 50) {
          gazeDwellBuffer.push({ element: currentGazeHoverLabel, duration: durationGaze });
          sidebarState.gazeDwells[currentGazeHoverLabel] = (sidebarState.gazeDwells[currentGazeHoverLabel] || 0) + durationGaze;
          currentGazeHoverStartTime = now;
        }
      }

      const data = {
        trail: trailBuffer,
        heatmap: heatmapBuffer,
        clicks: clickBuffer,
        dwells: dwellBuffer,
        gazeTrail: gazeTrailBuffer,
        gazeHeatmap: gazeHeatmapBuffer,
        gazeDwells: gazeDwellBuffer
      };

      trailBuffer = [];
      heatmapBuffer = [];
      clickBuffer = [];
      dwellBuffer = [];
      gazeTrailBuffer = [];
      gazeHeatmapBuffer = [];
      gazeDwellBuffer = [];

      return JSON.stringify(data);
    };

    // Start WebGaze socket client
    connectGazeWS();

    // Start Loops
    renderLoopId = requestAnimationFrame(renderLoop);
    
    // Refresh AOIs
    refreshAOIs();
    window.addEventListener('scroll', refreshAOIs, { passive: true });
    refreshIntervalId = setInterval(refreshAOIs, 400);

    // Sampling loop
    lastSampleTime = Date.now();
    sampleIntervalId = setInterval(() => {
      if (!isTrackingActive) return;
      const now = Date.now();
      const scrollX = window.scrollX;
      const scrollY = window.scrollY;

      // Mouse sampling
      if (lastClientX !== 0 || lastPageX !== 0) {
        const pageX = lastClientX + scrollX;
        const pageY = lastClientY + scrollY;
        const dt = now - lastSampleTime;

        if (dt <= 1000) {
          const dx = pageX - lastPageX;
          const dy = pageY - lastPageY;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist > 2) {
            trailBuffer.push({ x: pageX, y: pageY });
            visMousePoints.push({ x: pageX, y: pageY });
            sidebarState.mouseDataCount++;
            if (visMousePoints.length > 2000) visMousePoints.shift();
            lastPageX = pageX;
            lastPageY = pageY;
            isStationary = false;
            stationaryStart = 0;
          } else {
            if (!isStationary) {
              isStationary = true;
              stationaryStart = now;
            } else {
              const dwellDuration = now - stationaryStart;
              if (dwellDuration > DWELL_THRESHOLD) {
                heatmapBuffer.push({ x: pageX, y: pageY });
                visMousePoints.push({ x: pageX, y: pageY });
                sidebarState.mouseDataCount++;
                if (visMousePoints.length > 2000) visMousePoints.shift();
              }
            }
          }
        }
        lastSampleTime = now;
      }

      // Gaze sampling
      if (dot.x !== null && dot.y !== null) {
        const pageX = dot.x + scrollX;
        const pageY = dot.y + scrollY;
        const dtGaze = now - lastGazeSampleTime;

        if (dtGaze <= 1000) {
          const dx = pageX - lastGazePageX;
          const dy = pageY - lastGazePageY;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist > 6) {
            gazeTrailBuffer.push({ x: pageX, y: pageY });
            visGazePoints.push({ x: pageX, y: pageY });
            sidebarState.gazeDataCount++;
            if (visGazePoints.length > 2000) visGazePoints.shift();
            lastGazePageX = pageX;
            lastGazePageY = pageY;
            isGazeStationary = false;
            gazeStationaryStart = 0;
          } else {
            if (!isGazeStationary) {
              isGazeStationary = true;
              gazeStationaryStart = now;
            } else {
              const dwellDuration = now - gazeStationaryStart;
              if (dwellDuration > DWELL_THRESHOLD) {
                gazeHeatmapBuffer.push({ x: pageX, y: pageY });
                visGazePoints.push({ x: pageX, y: pageY });
                sidebarState.gazeDataCount++;
                if (visGazePoints.length > 2000) visGazePoints.shift();
              }
            }
          }
        }
        lastGazeSampleTime = now;
      }
    }, 50);

    sidebarIntervalId = setInterval(updateSidebarUI, 1000);
  }

  function stopInsightUXTracking() {
    if (!isTrackingActive) return;
    isTrackingActive = false;

    console.log('[InsightUX Guest] Stopping tracking session and cleaning overlays.');

    // Disconnect WS
    if (ws) {
      try { ws.close(); } catch(e) {}
      ws = null;
    }

    // Cancel frames and intervals
    if (renderLoopId) cancelAnimationFrame(renderLoopId);
    if (sampleIntervalId) clearInterval(sampleIntervalId);
    if (refreshIntervalId) clearInterval(refreshIntervalId);
    if (sidebarIntervalId) clearInterval(sidebarIntervalId);

    // Remove Event Listeners
    document.removeEventListener('mousemove', updateMousePos, true);
    document.removeEventListener('mouseenter', updateMousePos, true);
    document.removeEventListener('mouseover', updateMousePos, true);
    document.removeEventListener('click', handlePageClick, true);
    document.removeEventListener('keydown', handleKeyDown, true);
    window.removeEventListener('resize', resize);
    window.removeEventListener('resize', resizeCache);
    window.removeEventListener('scroll', refreshAOIs);

    delete window.insightuxAOIs;
    delete window.insightuxMouseData;

    // Remove nodes from DOM
    if (styleTag && styleTag.parentNode) styleTag.parentNode.removeChild(styleTag);
    if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
    if (sidebarContainer && sidebarContainer.parentNode) sidebarContainer.parentNode.removeChild(sidebarContainer);
    if (statusDiv && statusDiv.parentNode) statusDiv.parentNode.removeChild(statusDiv);

    // Reset parameters
    styleTag = null;
    canvas = null;
    ctx = null;
    sidebarContainer = null;
    statusDiv = null;

    visMousePoints.length = 0;
    visGazePoints.length = 0;
    visClicks.length = 0;
    trailBuffer.length = 0;
    heatmapBuffer.length = 0;
    clickBuffer.length = 0;
    dwellBuffer.length = 0;
    gazeTrailBuffer.length = 0;
    gazeHeatmapBuffer.length = 0;
    gazeDwellBuffer.length = 0;
  }

  // Register window custom event triggers (called by Electron preload script context)
  window.addEventListener('insightux-start', startInsightUXTracking);
  window.addEventListener('insightux-stop', stopInsightUXTracking);
})();
