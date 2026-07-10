"""
run_session.py
One command InsightUX session.

Instead of (start bridge, open Chrome, paste JS, go fullscreen, run tracker),
this hosts the website itself in a pywebview window, injects the AOI capture
plus a live gaze overlay automatically, and logs everything from Python.

Setup once:
    pip install pywebview

Run:
    python run_session.py https://www.yoursite.com
    (or just: python run_session.py  and it will prompt for a url)

Prerequisite:
    calibration.pkl must exist. If it does not, run calibrate.py first.

Writes:
    sessions/live/gaze_log.jsonl   raw gaze, every frame (for fixation detection)
    sessions/live/dom_log.jsonl    page elements + scroll + url, a few times a second

Smoothing:
    The on screen dot is smoothed with a One Euro filter (heavy smoothing when
    your eyes are still, light when they move fast). The LOGGED gaze stays raw,
    because Phase 3 fixation detection wants the unsmoothed signal.
"""

import os
import sys
import json
import time
import math
import threading
import asyncio
import websockets
from dataclasses import replace

import cv2
import numpy as np
import webview
import pyautogui

# =============================================================================
# GAZE WEBSOCKET SERVER (for streaming gaze to Chrome Extension)
# =============================================================================

class GazeWebSocketServer:
    def __init__(self, host="127.0.0.1", port=8765):
        self.host = host
        self.port = port
        self.connected = set()
        self.loop = None
        self._thread = None

    async def handler(self, websocket):
        self.connected.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.connected.remove(websocket)

    def start(self):
        self.loop = asyncio.new_event_loop()
        def run_loop():
            asyncio.set_event_loop(self.loop)
            async def main():
                async with websockets.serve(self.handler, self.host, self.port):
                    await asyncio.Future()  # run forever
            self.loop.run_until_complete(main())
            
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        print(f"[run_session] WebSocket server listening on ws://{self.host}:{self.port}")

    def broadcast_gaze(self, sx, sy):
        if not self.loop or not self.connected:
            return
        msg = json.dumps({"type": "gaze", "x": float(sx), "y": float(sy)})
        for ws in list(self.connected):
            asyncio.run_coroutine_threadsafe(ws.send(msg), self.loop)



from preprocessing.preprocessing_pipeline import (
    create_face_mesh,
    estimate_camera_matrix,
    estimate_head_pose,
    compute_iris_radius,
    step1_normalize,
    step2_illumination,
    LEFT_EYE_INDICES,
    LEFT_EAR_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_EAR_INDICES,
    RIGHT_IRIS_INDICES,
)
from inference_pipeline import InsightUXPipeline
from session_logger import GazeLogger


# =============================================================================
# CONFIG
# =============================================================================
def resolve_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. Try relative to CWD
    if os.path.exists(relative_path):
        return relative_path
    # 2. Try relative to script dir
    path_from_script = os.path.abspath(os.path.join(script_dir, relative_path))
    if os.path.exists(path_from_script):
        return path_from_script
    # 3. Try 2 levels up (if running from subfolders)
    path_from_root = os.path.abspath(os.path.join(script_dir, "..", "..", relative_path))
    if os.path.exists(path_from_root):
        return path_from_root
    # 4. Try 1 level up
    path_from_parent = os.path.abspath(os.path.join(script_dir, "..", relative_path))
    if os.path.exists(path_from_parent):
        return path_from_parent
    return relative_path

ONNX_PATH        = resolve_path("models/gaze_cnn_v4.onnx")
CALIBRATION_PATH = resolve_path("calibration.pkl")
SCREEN_W, SCREEN_H = pyautogui.size()

# Which eye patch to feed the CNN. MUST match calibrate.py, and recalibrate after changing.
# Settled from training history: the model was trained on the Step 2 "blended" patch
# (training ran apply_step2 on every MPIIGaze sample so train and inference match).
# So "blended" is correct. "norm_crop" is the mismatch that tracked badly.
PATCH_SOURCE   = "blended"

POSE_SMOOTH    = 0.65      # head pose EMA (0 = none, higher = smoother but laggier)
MAX_JUMP       = 220      # px, clamp wild teleport spikes
NO_FACE_RESET  = 15       # frames with no face before recentering
DOM_EVERY      = 6        # read page AOIs every N frames (about 5 per second)

# One Euro filter feel. Lower mincutoff = smoother when still.
# Lower beta = smoother but a touch laggier when the eyes move fast.
EURO_MINCUTOFF = 0.35
EURO_BETA      = 0.12

# FIX A / FIX B — MUST match calibrate.py and main_webcam_pipeline.py exactly.
# calibration.pkl was fit with pose scaled by /30 going into the model, and
# with the RBF trained on head-pitch-compensated pitch. Skipping either of
# these here would feed the calibrated model/RBF inputs it was never built
# for, even though the model and calibration themselves are now correct.
POSE_NORM_SCALE         = 30.0
HEAD_PITCH_COMPENSATION = 0.0

def normalize_pose(pitch_deg, yaw_deg, roll_deg):
    return np.array([
        pitch_deg / POSE_NORM_SCALE,
        yaw_deg   / POSE_NORM_SCALE,
        roll_deg  / POSE_NORM_SCALE,
    ], dtype=np.float32)

def compensate_pitch(raw_pitch, head_pitch_deg):
    return raw_pitch - np.radians(head_pitch_deg) * HEAD_PITCH_COMPENSATION

# FIX C — HEAD-YAW COMPENSATION (must match calibrate.py exactly)
HEAD_YAW_COMPENSATION = 0.0  # disabled

def compensate_yaw(raw_yaw, head_yaw_deg):
    return raw_yaw - np.radians(head_yaw_deg) * HEAD_YAW_COMPENSATION

SESSION_DIR = os.path.join("sessions", "live")
os.makedirs(SESSION_DIR, exist_ok=True)
DOM_PATH = os.path.join(SESSION_DIR, "dom_log.jsonl")


# =============================================================================
# ONE EURO FILTER (display smoothing only)
# =============================================================================

class OneEuroFilter:
    def __init__(self, mincutoff=0.8, beta=0.4, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-3
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


# =============================================================================
# JS injected into the page: AOI reader + gaze overlay
# =============================================================================

JS_SETUP = r"""
(function(){
  if (window.__insightux) { return; }
  const state = { aois: [] };
  const dwell = { pendLabel: null, pendSince: 0, activeLabel: null, emptySince: 0 };
  const DWELL_MS = 420;     // gaze must rest in a box this long before it lights up
  const RELEASE_MS = 1200;   // clear the highlight after gaze leaves everything this long

  let activeBox = null;
  const target  = { fx: 0.5, fy: 0.5 };
  const dot     = { x: null, y: null };
  const DOT_LERP = 0.07;   // lower = smoother/laggier, higher = snappier/twitchier

  // Sidebar state
  const sidebarState = {
    startTime: Date.now(),
    mouseDataCount: 0,
    gazeDataCount: 0,
    clicksCount: 0,
    clicksList: [],
    gazeDwells: {},
    mouseDwells: {}
  };

  // Main canvas for live rendering (gaze cursor, mouse cursor, trails, AOIs)
  const cv = document.createElement('canvas');
  cv.id = '__insightux_canvas';
  cv.style.cssText = 'position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483647;';
  (document.body || document.documentElement).appendChild(cv);
  const ctx = cv.getContext('2d');
  
  function resize(){ 
    cv.width = window.innerWidth; 
    cv.height = window.innerHeight; 
  }
  resize();
  window.addEventListener('resize', resize, {passive:true});

  // Inject Sidebar
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

  const sbContainer = document.createElement('div');
  sbContainer.id = '__insightux_sidebar_container';
  const styleTag = document.createElement('style');
  styleTag.textContent = css;
  document.head.appendChild(styleTag);

  sbContainer.innerHTML = `
    <div id="__insightux_sidebar_toggle"><span>›</span></div>
    <div id="__insightux_sidebar_panel">
      <div class="sidebar-header">
        <h2>InsightUX Analytics</h2>
        <div class="status-indicator">
          <span class="status-dot"></span>
          <span>Gaze Engine Active</span>
        </div>
      </div>
      <div class="sidebar-scroll">
        <!-- Timer & Counters Grid -->
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

        <!-- Controls Panel -->
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

        <!-- Dwell Interests -->
        <div class="section-panel">
          <div class="tabs-header">
            <button id="sb-tab-gaze" class="sb-tab-btn active">Gaze Dwell</button>
            <button id="sb-tab-mouse" class="sb-tab-btn">Mouse Hover</button>
          </div>

          <!-- Gaze tab content -->
          <div id="sb-content-gaze" class="sb-tab-content active">
            <div class="sb-hero-card" id="sb-gaze-hero" style="display:none;">
              <span class="hero-lbl">Most Looked Element</span>
              <span class="hero-el-name" id="sb-gaze-hero-el">-</span>
              <span class="hero-val" id="sb-gaze-hero-val">0.0s</span>
            </div>
            <div class="sb-interests-list" id="sb-gaze-list"></div>
          </div>

          <!-- Mouse tab content -->
          <div id="sb-content-mouse" class="sb-tab-content">
            <div class="sb-hero-card" id="sb-mouse-hero" style="display:none;">
              <span class="hero-lbl">Most Hovered Element</span>
              <span class="hero-el-name" id="sb-mouse-hero-el">-</span>
              <span class="hero-val" id="sb-mouse-hero-val">0.0s</span>
            </div>
            <div class="sb-interests-list" id="sb-mouse-list"></div>
          </div>
        </div>

        <!-- Clicks feed -->
        <div class="section-panel">
          <h3>Recent Clicks Log</h3>
          <div class="sb-click-feed" id="sb-click-feed-container">
            <div class="sb-no-clicks">No clicks recorded yet.</div>
          </div>
        </div>
      </div>
    </div>
  `;
  (document.body || document.documentElement).appendChild(sbContainer);

  let isCollapsed = false;
  function initSidebar() {
    const container = document.getElementById('__insightux_sidebar_container');
    const toggle = document.getElementById('__insightux_sidebar_toggle');
    const toggleArrow = toggle ? toggle.querySelector('span') : null;

    if (toggle && container) {
      toggle.addEventListener('click', () => {
        isCollapsed = !isCollapsed;
        container.classList.toggle('collapsed', isCollapsed);
        if (toggleArrow) {
          toggleArrow.textContent = isCollapsed ? '‹' : '›';
        }
      });
    }
  }

  function initTabs() {
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

  setInterval(updateSidebarUI, 1000);

  setInterval(function(){
    if (!document.body.contains(cv)) document.body.appendChild(cv);
    if (!document.body.contains(sbContainer)) document.body.appendChild(sbContainer);
  }, 1000);

  const MIN_W=40, MIN_H=24, MAX_AOIS=90, MAX_SCAN=2500;
  const PAD_PX = 90;
  const TALL_WRAPPER_H = 220;

  const ALWAYS_SELECTOR = "nav, header, footer, img, video, iframe, h1, h2, h3, button, figure, [class*='hero'], [class*='banner'], [class*='card']";
  const TEXT_CONTAINER_SELECTOR = "div, span, section, article, main, aside, p, li";

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

  function refresh(){
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

    state.aois = filtered.slice(0, MAX_AOIS).map(o => ({
      label:o.label, x:o.x, y:o.y, w:o.w, h:o.h, sticky:o.sticky
    }));
  }
  refresh();
  window.addEventListener('scroll', refresh, {passive:true});
  setInterval(refresh, 400);

  window.insightuxAOIs = function(){
    return JSON.stringify({
      url: location.href,
      scrollX: Math.round(window.scrollX),
      scrollY: Math.round(window.scrollY),
      viewport:{w:window.innerWidth,h:window.innerHeight},
      page:{w:document.documentElement.scrollWidth,h:document.documentElement.scrollHeight},
      aois: state.aois
    });
  };

  window.insightuxUpdate = function(fx, fy){
    target.fx = fx;
    target.fy = fy;
  };

  // ─── MOUSE AND GAZE TRACKING logic (adapted from Chrome Extension) ───
  let isTracking = true;
  let showGazeCursor = true;
  let showMouseCursor = true;
  let showTrails = true;
  let showHeatmap = false;
  let heatmapType = "combined"; // "mouse", "gaze", or "combined"

  let lastClientX = 0, lastClientY = 0;
  let lastPageX = 0, lastPageY = 0;
  let lastSampleTime = Date.now();
  let stationaryStart = 0;
  let isStationary = false;
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

  function handleHoverChange(newLabel) {
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

  document.addEventListener('mousemove', updateMousePos, true);
  document.addEventListener('mouseenter', updateMousePos, true);
  document.addEventListener('mouseover', updateMousePos, true);
  document.addEventListener('mouseleave', () => {
    lastClientX = 0; lastClientY = 0;
    handleHoverChange(null);
  }, true);

  document.addEventListener('click', (e) => {
    if (e.target === cv) return;
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

    // Update sidebar state
    sidebarState.clicksCount++;
    sidebarState.clicksList.unshift(logEntry);
    if (sidebarState.clicksList.length > 30) sidebarState.clicksList.pop();

    if (showHeatmap) {
      updateHeatmapCache();
    }
  }, true);

  // Periodical sampling loop (every 50ms)
  let lastGazePageX = 0, lastGazePageY = 0;
  let lastGazeSampleTime = Date.now();
  let gazeStationaryStart = 0;
  let isGazeStationary = false;

  setInterval(() => {
    const now = Date.now();
    const scrollX = window.scrollX;
    const scrollY = window.scrollY;

    // ─── 1. MOUSE SAMPLING ───
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

    // ─── 2. GAZE SAMPLING ───
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

  // Return mouse data buffer to Python and clear local state
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

  // ─── Heatmap rendering on offscreen canvas for top performance ───
  const cacheCanvas = document.createElement('canvas');
  const cacheCtx = cacheCanvas.getContext('2d');
  
  function resizeCache() {
    cacheCanvas.width = window.innerWidth;
    cacheCanvas.height = window.innerHeight;
  }
  resizeCache();
  window.addEventListener('resize', resizeCache, {passive:true});

  const brushRadius = 55;
  const brushCanvas = document.createElement('canvas');
  brushCanvas.width = brushRadius * 2;
  brushCanvas.height = brushRadius * 2;
  const brushCtx = brushCanvas.getContext('2d');
  const brushGrad = brushCtx.createRadialGradient(brushRadius, brushRadius, 0, brushRadius, brushRadius, brushRadius);
  brushGrad.addColorStop(0, 'rgba(0,0,0,0.06)');
  brushGrad.addColorStop(1, 'rgba(0,0,0,0)');
  brushCtx.fillStyle = brushGrad;
  brushCtx.fillRect(0, 0, brushRadius * 2, brushRadius * 2);

  const gradientCanvas = document.createElement('canvas');
  gradientCanvas.width = 256;
  gradientCanvas.height = 1;
  const gradCtx = gradientCanvas.getContext('2d');
  const grad = gradCtx.createLinearGradient(0, 0, 256, 0);
  grad.addColorStop(0.0, 'rgba(0, 0, 255, 0)');     
  grad.addColorStop(0.1, 'rgba(0, 0, 255, 1)');     // Blue
  grad.addColorStop(0.4, 'rgba(0, 255, 255, 1)');   // Cyan
  grad.addColorStop(0.6, 'rgba(0, 255, 0, 1)');     // Green
  grad.addColorStop(0.8, 'rgba(255, 255, 0, 1)');   // Yellow
  grad.addColorStop(1.0, 'rgba(255, 0, 0, 1)');     // Red
  gradCtx.fillStyle = grad;
  gradCtx.fillRect(0, 0, 256, 1);
  const gradientMap = gradCtx.getImageData(0, 0, 256, 1).data;

  function updateHeatmapCache() {
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
          pix[i]     = gradientMap[cOffset];     // R
          pix[i + 1] = gradientMap[cOffset + 1]; // G
          pix[i + 2] = gradientMap[cOffset + 2]; // B
          pix[i + 3] = Math.min(255, 140 + a);    // Alpha boost
        }
      }
      cacheCtx.putImageData(imgData, 0, 0);
    } catch(e) {
      console.error("Heatmap rendering failed:", e);
    }
  }

  setInterval(() => {
    if (showHeatmap && isTracking) {
      updateHeatmapCache();
    }
  }, 500);

  // Initialize UI controls
  function initControls() {
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

  initSidebar();
  initTabs();
  initControls();

  // ─── Keybind listeners for controls ───
  document.addEventListener('keydown', function(e){
    if (e.key==='Escape' && window.pywebview && window.pywebview.api){
      window.pywebview.api.stop();
    }
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
  });

  let statusTimeout = null;
  const statusDiv = document.createElement('div');
  statusDiv.style.cssText = 'position:fixed;top:20px;right:20px;background:rgba(0,0,0,0.85);color:#fff;padding:8px 16px;border-radius:4px;font-family:Arial,sans-serif;font-size:14px;z-index:2147483647;pointer-events:none;display:none;border:1px solid #7B2FBE;';
  document.body.appendChild(statusDiv);

  function showStatusIndicator(msg) {
    statusDiv.textContent = msg;
    statusDiv.style.display = 'block';
    if (statusTimeout) clearTimeout(statusTimeout);
    statusTimeout = setTimeout(() => {
      statusDiv.style.display = 'none';
    }, 1500);
  }

  function renderLoop(){
    const w = window.innerWidth, h = window.innerHeight;
    const tx = target.fx * w, ty = target.fy * h;
    if (dot.x === null){ dot.x = tx; dot.y = ty; }
    dot.x += (tx - dot.x) * DOT_LERP;
    dot.y += (ty - dot.y) * DOT_LERP;

    const now = performance.now();
    const px = dot.x, py = dot.y;
    const scrollX = window.scrollX;
    const scrollY = window.scrollY;

    // Gaze box candidate
    let cand = null;
    for (const a of state.aois){
      if (px>=a.x-PAD_PX && px<=a.x+a.w+PAD_PX && py>=a.y-PAD_PX && py<=a.y+a.h+PAD_PX){
        if (!cand || (a.w*a.h)<(cand.w*cand.h)) cand = a;
      }
    }

    // Dwell + hysteresis
    const candLabel = cand ? cand.label : null;
    if (candLabel !== dwell.pendLabel){ dwell.pendLabel = candLabel; dwell.pendSince = now; }
    if (cand){
      dwell.emptySince = 0;
      if (dwell.activeLabel !== cand.label && (now - dwell.pendSince) >= DWELL_MS){
        dwell.activeLabel = cand.label;
      }
    } else {
      if (dwell.emptySince === 0) dwell.emptySince = now;
      if (dwell.activeLabel && (now - dwell.emptySince) >= RELEASE_MS){
        dwell.activeLabel = null;
      }
    }

    let active = null;
    if (dwell.activeLabel){
      for (const a of state.aois){ if (a.label === dwell.activeLabel){ active = a; break; } }
      if (!active) dwell.activeLabel = null;
    }
    activeBox = active;

    // Update Gaze Hover label
    handleGazeHoverChange(activeBox ? activeBox.label : null);

    ctx.clearRect(0,0,cv.width,cv.height);

    // 1. Draw Heatmap (if enabled)
    if (showHeatmap) {
      ctx.drawImage(cacheCanvas, 0, 0);
    }

    // 2. Draw Clicks & click-to-gaze lines
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

    // 3. Draw Trails (if enabled)
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
    if (activeBox){
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

    requestAnimationFrame(renderLoop);
  }
  requestAnimationFrame(renderLoop);

  window.__insightux = true;
})();
"""


# =============================================================================
# Control object exposed to JS (so Escape can stop the session)
# =============================================================================

class Api:
    def __init__(self):
        self.running = True

    def stop(self):
        self.running = False
        return True


api = Api()


# =============================================================================
# GAZE WORKER (runs in a pywebview background thread)
# =============================================================================

def inject(window):
    """(Re)inject the overlay + AOI script. Called on every page load."""
    try:
        window.evaluate_js(JS_SETUP)
    except Exception as e:
        print(f"[run_session] inject failed (will retry): {e}")


def gaze_worker(window):
    if not os.path.exists(CALIBRATION_PATH):
        print(f"[run_session] No {CALIBRATION_PATH}. Run calibrate.py first.")
        api.running = False
        window.destroy()
        return

    pipeline  = InsightUXPipeline(ONNX_PATH, CALIBRATION_PATH)
    face_mesh = create_face_mesh(static_image_mode=False)
    cap       = cv2.VideoCapture(0)
    logger    = GazeLogger(SCREEN_W, SCREEN_H, session_dir=SESSION_DIR)
    dom_f     = open(DOM_PATH, "w", buffering=1)
    
    # Open Mouse Log
    mouse_log_path = os.path.join(SESSION_DIR, "mouse_log.jsonl")
    mouse_f = open(mouse_log_path, "w", buffering=1)
    print(f"[run_session] Writing mouse tracking logs to {mouse_log_path}")

    # Start Gaze WebSocket Server
    ws_server = GazeWebSocketServer()
    ws_server.start()

    # re-inject on each navigation, and inject the first page now
    window.events.loaded += lambda: inject(window)
    for _ in range(10):
        inject(window)
        try:
            if window.evaluate_js("window.__insightux === true"):
                break
        except Exception:
            pass
        time.sleep(0.4)

    print("[run_session] tracking. Look at the site. Press Esc to stop.")
    print("[run_session] Hotkeys: H = Heatmap | T = Trails | C = Gaze Cursor | M = Mouse Tracker | Y = Heatmap Mode")

    cam_matrix    = None
    last_x        = SCREEN_W / 2
    last_y        = SCREEN_H / 2
    fil_x         = OneEuroFilter(mincutoff=EURO_MINCUTOFF, beta=EURO_BETA)
    fil_y         = OneEuroFilter(mincutoff=EURO_MINCUTOFF, beta=EURO_BETA)
    sm_pitch = sm_yaw = sm_roll = None
    no_face_count = 0
    frame_i       = 0

    while api.running:
        ret, frame = cap.read()
        if not ret:
            break

        if cam_matrix is None:
            cam_matrix = estimate_camera_matrix(frame.shape)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            no_face_count += 1
            if no_face_count >= NO_FACE_RESET:
                last_x, last_y = SCREEN_W / 2, SCREEN_H / 2
                fil_x.reset(); fil_y.reset()
                sm_pitch = sm_yaw = sm_roll = None
                no_face_count = 0
            continue
        no_face_count = 0

        lms       = results.multi_face_landmarks[0].landmark
        head_pose = estimate_head_pose(lms, frame.shape, cam_matrix)
        if head_pose is None:
            continue

        # smooth the head pose: cuts solvePnP jitter feeding both the warp and the model
        if sm_pitch is None:
            sm_pitch, sm_yaw, sm_roll = head_pose.pitch, head_pose.yaw, head_pose.roll
        else:
            b = POSE_SMOOTH
            sm_pitch = b*sm_pitch + (1-b)*head_pose.pitch
            sm_yaw   = b*sm_yaw   + (1-b)*head_pose.yaw
            sm_roll  = b*sm_roll  + (1-b)*head_pose.roll
        head_pose = replace(head_pose, pitch=sm_pitch, yaw=sm_yaw, roll=sm_roll)

        pose_vec = normalize_pose(sm_pitch, sm_yaw, sm_roll)   # FIX A

        def get_patch(eye_idx, ear_idx, iris_idx):
            s1 = step1_normalize(frame, lms, head_pose, eye_idx, ear_idx, iris_idx)
            if not s1.is_open:
                return None
            if PATCH_SOURCE == "norm":
                return s1.norm_crop
            ir = compute_iris_radius(lms, iris_idx, frame.shape)
            s2 = step2_illumination(s1, ir)
            return s2.blended if s2.is_usable else None

        left_patch  = get_patch(LEFT_EYE_INDICES,  LEFT_EAR_INDICES,  LEFT_IRIS_INDICES)
        right_patch = get_patch(RIGHT_EYE_INDICES, RIGHT_EAR_INDICES, RIGHT_IRIS_INDICES)
        if left_patch is None and right_patch is None:
            continue
        if left_patch  is None: left_patch  = right_patch
        if right_patch is None: right_patch = left_patch

        _, _, raw_pitch, raw_yaw = pipeline.predict_gaze_vector(left_patch, pose_vec, right_patch)
        pitch = compensate_pitch(raw_pitch, sm_pitch)   # FIX B
        yaw   = compensate_yaw(raw_yaw, sm_yaw)         # FIX C
        sx, sy = pipeline.calibration.predict(pitch, yaw)
        sx = max(0.0, min(sx, SCREEN_W))
        sy = max(0.0, min(sy, SCREEN_H))

        # guard against wild teleport spikes
        jump = np.hypot(sx - last_x, sy - last_y)
        if jump > MAX_JUMP:
            sx = last_x + (sx - last_x) * 0.3
            sy = last_y + (sy - last_y) * 0.3
        last_x, last_y = sx, sy

        # log the RAW gaze (Phase 3 fixation detection wants it unsmoothed)
        logger.log(sx, sy)

        # Broadcast gaze to all connected WebSocket clients (e.g. Chrome Extension)
        ws_server.broadcast_gaze(sx, sy)

        # smooth only the DISPLAYED dot with a One Euro filter
        now = time.time()
        fx = fil_x(sx / SCREEN_W, now)
        fy = fil_y(sy / SCREEN_H, now)
        try:
            window.evaluate_js(f"window.insightuxUpdate({fx:.5f},{fy:.5f})")
        except Exception:
            pass

        # log the page state a few times a second
        frame_i += 1
        if frame_i % DOM_EVERY == 0:
            # 1. Log DOM elements under gaze
            try:
                raw = window.evaluate_js("window.insightuxAOIs()")
                if raw:
                    rec = json.loads(raw)
                    rec["type"] = "dom"
                    rec["t"] = round(time.time(), 4)
                    dom_f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

            # 2. Log Mouse tracking details (trail, clicks, dwells)
            try:
                raw_mouse = window.evaluate_js("window.insightuxMouseData()")
                if raw_mouse:
                    mouse_data = json.loads(raw_mouse)
                    t_now = round(time.time(), 4)
                    
                    # Log clicks
                    for click in mouse_data.get("clicks", []):
                        click["type"] = "click"
                        click["t"] = t_now
                        mouse_f.write(json.dumps(click) + "\n")
                        
                    # Log mouse element hover dwells
                    for dwell_item in mouse_data.get("dwells", []):
                        dwell_item["type"] = "mouse_dwell"
                        dwell_item["t"] = t_now
                        mouse_f.write(json.dumps(dwell_item) + "\n")
                        
                    # Log gaze element hover dwells
                    for gaze_dwell_item in mouse_data.get("gazeDwells", []):
                        gaze_dwell_item["type"] = "gaze_dwell"
                        gaze_dwell_item["t"] = t_now
                        mouse_f.write(json.dumps(gaze_dwell_item) + "\n")

                    # Log mouse trail segments
                    trail = mouse_data.get("trail")
                    if trail:
                        mouse_f.write(json.dumps({
                            "type": "mouse_trail",
                            "t": t_now,
                            "points": trail
                        }) + "\n")

                    # Log gaze trail segments
                    gaze_trail = mouse_data.get("gazeTrail")
                    if gaze_trail:
                        mouse_f.write(json.dumps({
                            "type": "gaze_trail",
                            "t": t_now,
                            "points": gaze_trail
                        }) + "\n")
            except Exception as e:
                print(f"[run_session] Error logging mouse data: {e}")

        time.sleep(0.005)

    cap.release()
    logger.close()
    dom_f.close()
    mouse_f.close()
    print(f"[run_session] session saved in {SESSION_DIR}")
    try:
        window.destroy()
    except Exception:
        pass
    try:
        window.destroy()
    except Exception:
        pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("Website url to test: ").strip()
    if not url.startswith("http"):
        url = "https://" + url

    window = webview.create_window("InsightUX", url, fullscreen=True, js_api=api)
    webview.start(gaze_worker, window)