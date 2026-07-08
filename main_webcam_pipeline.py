import cv2
import numpy as np
import threading
import tkinter as tk
import sys
import ctypes

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


# =============================================================================
# FIX A — POSE NORMALIZATION  (must match calibrate.py exactly)
# =============================================================================
POSE_NORM_SCALE = 30.0

def normalize_pose(head_pose):
    return np.array([
        head_pose.pitch / POSE_NORM_SCALE,
        head_pose.yaw   / POSE_NORM_SCALE,
        head_pose.roll  / POSE_NORM_SCALE,
    ], dtype=np.float32)


# =============================================================================
# FIX B — HEAD-PITCH COMPENSATION  (must match calibrate.py exactly)
# =============================================================================
HEAD_PITCH_COMPENSATION = 0.4

def compensate_pitch(raw_pitch, head_pitch_deg):
    return raw_pitch - np.radians(head_pitch_deg) * HEAD_PITCH_COMPENSATION


# =============================================================================
# FIX C — HEAD-YAW COMPENSATION  (must match calibrate.py exactly)
# =============================================================================
HEAD_YAW_COMPENSATION = 0.0  # disabled - unvalidated guess, reverted

def compensate_yaw(raw_yaw, head_yaw_deg):
    return raw_yaw - np.radians(head_yaw_deg) * HEAD_YAW_COMPENSATION


# =============================================================================
# PIPELINE + FACE MESH
# =============================================================================
def resolve_path(relative_path):
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. Try relative to CWD
    if os.path.exists(relative_path):
        return relative_path
    # 2. Try relative to script dir
    path_from_script = os.path.abspath(os.path.join(script_dir, relative_path))
    if os.path.exists(path_from_script):
        return path_from_script
    # 3. Try 2 levels up
    path_from_root = os.path.abspath(os.path.join(script_dir, "..", "..", relative_path))
    if os.path.exists(path_from_root):
        return path_from_root
    # 4. Try 1 level up
    path_from_parent = os.path.abspath(os.path.join(script_dir, "..", relative_path))
    if os.path.exists(path_from_parent):
        return path_from_parent
    return relative_path

pipeline = InsightUXPipeline(
    resolve_path("models/gaze_cnn_v4.onnx"),
    resolve_path("calibration.pkl")
)

face_mesh = create_face_mesh(static_image_mode=False)
cap       = cv2.VideoCapture(0)

cam_matrix    = None
smooth_x      = 0.0
smooth_y      = 0.0
alpha         = 0.70
no_face_count = 0
NO_FACE_RESET = 15
MAX_JUMP      = 350

SCREEN_W = 1493
SCREEN_H = 933

smooth_x = SCREEN_W / 2
smooth_y = SCREEN_H / 2

# 3x3 grid (was 4x4). Fewer, bigger zones - matches what validate.py
# actually measured well (67% hit rate on a 3x3 grid), and bigger zones
# are more forgiving of the ~110-150px average gaze error than a finer
# 4x4 grid was.
COL_BOUNDS = [0, SCREEN_W // 3, 2 * SCREEN_W // 3, SCREEN_W]
ROW_BOUNDS = [0, SCREEN_H // 3, 2 * SCREEN_H // 3, SCREEN_H]

REGION_NAMES = [
    ["TOP-LEFT",    "TOP-CENTER",    "TOP-RIGHT"],
    ["MID-LEFT",    "CENTER",        "MID-RIGHT"],
    ["BOTTOM-LEFT", "BOTTOM-CENTER", "BOTTOM-RIGHT"],
]

gaze_state = {
    "row": 1, "col": 1,
    "sx": SCREEN_W // 2,
    "sy": SCREEN_H // 2,
    "active": True
}
state_lock = threading.Lock()


def get_region(sx, sy):
    col = 0
    for i in range(1, 3):
        if sx >= COL_BOUNDS[i]:
            col = i
    row = 0
    for i in range(1, 3):
        if sy >= ROW_BOUNDS[i]:
            row = i
    return row, col


# =============================================================================
# TKINTER TRANSPARENT OVERLAY
# =============================================================================

class GazeOverlay:

    GRID_COLOR   = '#7B2FBE'
    ACTIVE_COLOR = '#9B59FF'
    TEXT_COLOR   = '#FFFFFF'

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("InsightUX Overlay")
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 1.0)
        self.root.configure(bg='black')
        self.root.wm_attributes('-transparentcolor', 'black')

        self.W = self.root.winfo_screenwidth()
        self.H = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(
            self.root, bg='black',
            highlightthickness=0,
            width=self.W, height=self.H
        )
        self.canvas.pack(fill='both', expand=True)

        self.col_b = [0, self.W // 3, 2 * self.W // 3, self.W]
        self.row_b = [0, self.H // 3, 2 * self.H // 3, self.H]

        self.root.after(600, self._apply_clickthrough)

        try:
            import keyboard
            keyboard.add_hotkey('esc', self._quit)
            keyboard.add_hotkey('r',   self._reset)
        except ImportError:
            self.root.bind('<Escape>', lambda e: self._quit())
            self.root.bind('r',        lambda e: self._reset())
            print("[Overlay] tip: pip install keyboard for global ESC support")

    def _apply_clickthrough(self):
        if sys.platform == 'win32':
            hwnd  = ctypes.windll.user32.FindWindowW(None, "InsightUX Overlay")
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, -20, style | 0x80000 | 0x20
            )

    def _quit(self):
        with state_lock:
            gaze_state["active"] = False
        try:
            self.root.after(0, self.root.destroy)
        except Exception:
            pass

    def _reset(self):
        global smooth_x, smooth_y
        smooth_x = SCREEN_W / 2
        smooth_y = SCREEN_H / 2

    def _draw(self):
        if not gaze_state["active"]:
            return

        with state_lock:
            row = gaze_state["row"]
            col = gaze_state["col"]

        self.canvas.delete("all")

        for r in range(3):
            for c in range(3):
                x1 = self.col_b[c]
                x2 = self.col_b[c + 1]
                y1 = self.row_b[r]
                y2 = self.row_b[r + 1]
                is_active = (r == row and c == col)

                if is_active:
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill='#4B0082', stipple='gray25', outline=''
                    )
                    self.canvas.create_rectangle(
                        x1+2, y1+2, x2-2, y2-2,
                        fill='', outline=self.ACTIVE_COLOR, width=3
                    )
                    cx_reg = (x1 + x2) // 2
                    cy_reg = (y1 + y2) // 2
                    self.canvas.create_text(
                        cx_reg, cy_reg,
                        text=REGION_NAMES[r][c],
                        fill=self.TEXT_COLOR,
                        font=('Arial', 18, 'bold')
                    )
                else:
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill='', outline=self.GRID_COLOR, width=1
                    )

        self.canvas.create_text(
            20, 20,
            text=f"Looking at: {REGION_NAMES[row][col]}",
            anchor='nw', fill='#CC99FF',
            font=('Arial', 13, 'bold')
        )
        self.canvas.create_text(
            20, 42,
            text="ESC = quit   R = reset",
            anchor='nw', fill='#555555',
            font=('Arial', 10)
        )

        self.root.after(33, self._draw)

    def run(self):
        self._draw()
        self.root.mainloop()


# =============================================================================
# GAZE TRACKING THREAD
# =============================================================================

def gaze_loop():
    global smooth_x, smooth_y, no_face_count, cam_matrix

    print("Gaze tracking started.")
    print("ESC to quit  |  R to reset gaze")

    while True:
        with state_lock:
            if not gaze_state["active"]:
                break

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
                smooth_x = SCREEN_W / 2
                smooth_y = SCREEN_H / 2
                no_face_count = 0
            continue

        no_face_count = 0
        lms       = results.multi_face_landmarks[0].landmark
        head_pose = estimate_head_pose(lms, frame.shape, cam_matrix)
        if head_pose is None:
            continue

        # FIX A: normalize pose to [-1, +1]
        head_pose_vec = normalize_pose(head_pose)

        def get_patch(eye_idx, ear_idx, iris_idx):
            s1 = step1_normalize(frame, lms, head_pose, eye_idx, ear_idx, iris_idx)
            if not s1.is_open:
                return None
            ir = compute_iris_radius(lms, iris_idx, frame.shape)
            s2 = step2_illumination(s1, ir)
            return s2.blended if s2.is_usable else None

        left_patch  = get_patch(LEFT_EYE_INDICES,  LEFT_EAR_INDICES,  LEFT_IRIS_INDICES)
        right_patch = get_patch(RIGHT_EYE_INDICES, RIGHT_EAR_INDICES, RIGHT_IRIS_INDICES)

        if left_patch is None and right_patch is None:
            continue
        if left_patch  is None: left_patch  = right_patch
        if right_patch is None: right_patch = left_patch

        _, _, raw_pitch, raw_yaw = pipeline.predict_gaze_vector(
            left_patch, head_pose_vec, right_patch
        )

        # FIX B / FIX C: remove head-pitch and head-yaw contribution before RBF lookup
        avg_pitch = compensate_pitch(raw_pitch, head_pose.pitch)
        avg_yaw   = compensate_yaw(raw_yaw, head_pose.yaw)

        sx, sy = pipeline.calibration.predict(avg_pitch, avg_yaw)
        sx = max(0.0, min(sx, SCREEN_W))
        sy = max(0.0, min(sy, SCREEN_H))

        jump = np.sqrt((sx - smooth_x)**2 + (sy - smooth_y)**2)
        if jump > MAX_JUMP:
            sx = smooth_x + (sx - smooth_x) * 0.3
            sy = smooth_y + (sy - smooth_y) * 0.3

        smooth_x = alpha * smooth_x + (1 - alpha) * sx
        smooth_y = alpha * smooth_y + (1 - alpha) * sy

        row, col = get_region(smooth_x, smooth_y)

        with state_lock:
            gaze_state["row"] = row
            gaze_state["col"] = col
            gaze_state["sx"]  = int(smooth_x)
            gaze_state["sy"]  = int(smooth_y)

    cap.release()


# =============================================================================
if __name__ == "__main__":
    t = threading.Thread(target=gaze_loop, daemon=True)
    t.start()
    overlay = GazeOverlay()
    overlay.run()