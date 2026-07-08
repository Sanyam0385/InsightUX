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
import pyautogui

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
    # 3. Try 2 levels up (if running from MY_WEBGazer/python_backend)
    path_from_root = os.path.abspath(os.path.join(script_dir, "..", "..", relative_path))
    if os.path.exists(path_from_root):
        return path_from_root
    # 4. Try 1 level up (if running from MY_WEBGazer)
    path_from_parent = os.path.abspath(os.path.join(script_dir, "..", relative_path))
    if os.path.exists(path_from_parent):
        return path_from_parent
    return relative_path

ONNX_PATH        = resolve_path("models/gaze_cnn_v4.onnx")
CALIBRATION_PATH = resolve_path("calibration.pkl")
SCREEN_W, SCREEN_H = pyautogui.size()
PATCH_SOURCE   = "blended"
POSE_SMOOTH    = 0.65      # head pose EMA
MAX_JUMP       = 220      # px, clamp wild teleport spikes
NO_FACE_RESET  = 15       # frames with no face before recentering

POSE_NORM_SCALE         = 30.0
HEAD_PITCH_COMPENSATION = 0.0
HEAD_YAW_COMPENSATION   = 0.0

def normalize_pose(pitch_deg, yaw_deg, roll_deg):
    return np.array([
        pitch_deg / POSE_NORM_SCALE,
        yaw_deg   / POSE_NORM_SCALE,
        roll_deg  / POSE_NORM_SCALE,
    ], dtype=np.float32)

def compensate_pitch(raw_pitch, head_pitch_deg):
    return raw_pitch - np.radians(head_pitch_deg) * HEAD_PITCH_COMPENSATION

def compensate_yaw(raw_yaw, head_yaw_deg):
    return raw_yaw - np.radians(head_yaw_deg) * HEAD_YAW_COMPENSATION

# =============================================================================
# GAZE WEBSOCKET SERVER
# =============================================================================
class GazeWebSocketServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.connected = set()
        self.loop = None
        self._thread = None
        self._server = None

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
                async with websockets.serve(self.handler, self.host, self.port) as server:
                    self._server = server
                    await asyncio.Future()  # run forever
            try:
                self.loop.run_until_complete(main())
            except Exception as e:
                print(f"[gaze_server] WebSocket Server Exception: {e}", file=sys.stderr)
            
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        print(f"[gaze_server] WebSocket server listening on ws://{self.host}:{self.port}")

    def broadcast_gaze(self, sx, sy):
        if not self.loop or not self.connected:
            return
        msg = json.dumps({"type": "gaze", "x": float(sx), "y": float(sy)})
        for ws in list(self.connected):
            asyncio.run_coroutine_threadsafe(ws.send(msg), self.loop)

# =============================================================================
# MAIN TRACKING PIPELINE
# =============================================================================
def main():
    if not os.path.exists(CALIBRATION_PATH):
        print(f"[gaze_server] Error: Calibration file {CALIBRATION_PATH} not found. Please calibrate first.")
        sys.exit(1)

    print("[gaze_server] Loading gaze estimation models...")
    pipeline  = InsightUXPipeline(ONNX_PATH, CALIBRATION_PATH)
    face_mesh = create_face_mesh(static_image_mode=False)
    cap       = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[gaze_server] Error: Webcam could not be opened.", file=sys.stderr)
        sys.exit(1)

    ws_server = GazeWebSocketServer()
    ws_server.start()

    session_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("sessions", "live")
    os.makedirs(session_dir, exist_ok=True)
    gaze_logger = GazeLogger(SCREEN_W, SCREEN_H, session_dir=session_dir)

    print("[gaze_server] Tracking active. Press 'q' in the camera window to stop.")
    
    cam_matrix    = None
    last_x        = SCREEN_W / 2
    last_y        = SCREEN_H / 2
    sm_pitch = sm_yaw = sm_roll = None
    no_face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[gaze_server] Error: Failed to read frame from webcam.", file=sys.stderr)
            break

        if cam_matrix is None:
            cam_matrix = estimate_camera_matrix(frame.shape)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            no_face_count += 1
            if no_face_count >= NO_FACE_RESET:
                last_x, last_y = SCREEN_W / 2, SCREEN_H / 2
                sm_pitch = sm_yaw = sm_roll = None
                no_face_count = 0
            
            # Show standard frame on no face
            cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("InsightUX Gaze Server Preview", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

        no_face_count = 0
        lms       = results.multi_face_landmarks[0].landmark
        head_pose = estimate_head_pose(lms, frame.shape, cam_matrix)
        
        if head_pose is None:
            cv2.imshow("InsightUX Gaze Server Preview", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

        # smooth the head pose
        if sm_pitch is None:
            sm_pitch, sm_yaw, sm_roll = head_pose.pitch, head_pose.yaw, head_pose.roll
        else:
            b = POSE_SMOOTH
            sm_pitch = b*sm_pitch + (1-b)*head_pose.pitch
            sm_yaw   = b*sm_yaw   + (1-b)*head_pose.yaw
            sm_roll  = b*sm_roll  + (1-b)*head_pose.roll
        head_pose = replace(head_pose, pitch=sm_pitch, yaw=sm_yaw, roll=sm_roll)

        pose_vec = normalize_pose(sm_pitch, sm_yaw, sm_roll)

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
            cv2.imshow("InsightUX Gaze Server Preview", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue
            
        if left_patch  is None: left_patch  = right_patch
        if right_patch is None: right_patch = left_patch

        _, _, raw_pitch, raw_yaw = pipeline.predict_gaze_vector(left_patch, pose_vec, right_patch)
        pitch = compensate_pitch(raw_pitch, sm_pitch)
        yaw   = compensate_yaw(raw_yaw, sm_yaw)
        sx, sy = pipeline.calibration.predict(pitch, yaw)
        sx = max(0.0, min(sx, SCREEN_W))
        sy = max(0.0, min(sy, SCREEN_H))

        # guard against wild teleport spikes
        jump = np.hypot(sx - last_x, sy - last_y)
        if jump > MAX_JUMP:
            sx = last_x + (sx - last_x) * 0.3
            sy = last_y + (sy - last_y) * 0.3
        last_x, last_y = sx, sy

        # Broadcast raw coordinate over Websocket
        ws_server.broadcast_gaze(sx, sy)

        # Log raw gaze coordinate to sessions/live/gaze_log.jsonl
        gaze_logger.log(sx, sy)

        # Draw visual feedback indicator on frame
        cv2.putText(frame, f"Gaze: {int(sx)}, {int(sy)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw circles around irises for verification
        h_img, w_img, _ = frame.shape
        for idx in LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES:
            pt = lms[idx]
            cx, cy = int(pt.x * w_img), int(pt.y * h_img)
            cv2.circle(frame, (cx, cy), 1, (255, 0, 255), -1)

        cv2.imshow("InsightUX Gaze Server Preview", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    gaze_logger.close()
    cv2.destroyAllWindows()
    print("[gaze_server] Gaze server closed cleanly.")

if __name__ == "__main__":
    main()
