"""
session_logger.py
Phase 1: logs the live gaze stream to a per session file.

Writes one line of JSON per frame to:
    sessions/live/gaze_log.jsonl

Fields per line:
    t       wall clock seconds (time.time()). Same epoch the browser uses
            via Date.now()/1000, so gaze and DOM logs align by time.
    sx, sy  gaze in calibration screen space (SCREEN_W x SCREEN_H)
    fx, fy  gaze as a fraction of the screen (0..1). Phase 4 multiplies
            these by the browser viewport size to land on a real element.

The file is truncated on each run, so one tracker run equals one clean session.
"""

import os
import json
import time


class GazeLogger:
    def __init__(self, screen_w, screen_h, session_dir=os.path.join("sessions", "live")):
        self.screen_w = float(screen_w)
        self.screen_h = float(screen_h)

        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.gaze_path = os.path.join(session_dir, "gaze_log.jsonl")

        # "w" truncates: each run starts a fresh gaze log
        self._f = open(self.gaze_path, "w", buffering=1)

        meta = {
            "type": "meta",
            "screen_w": screen_w,
            "screen_h": screen_h,
            "t": round(time.time(), 4),
        }
        self._f.write(json.dumps(meta) + "\n")
        print(f"[GazeLogger] writing gaze stream to {self.gaze_path}")

    def log(self, sx, sy, t=None):
        if t is None:
            t = time.time()
        fx = sx / self.screen_w if self.screen_w else 0.0
        fy = sy / self.screen_h if self.screen_h else 0.0
        rec = {
            "type": "gaze",
            "t": round(t, 4),
            "sx": round(float(sx), 2),
            "sy": round(float(sy), 2),
            "fx": round(float(fx), 5),
            "fy": round(float(fy), 5),
        }
        self._f.write(json.dumps(rec) + "\n")

    def close(self):
        try:
            self._f.close()
            print(f"[GazeLogger] gaze log closed: {self.gaze_path}")
        except Exception:
            pass