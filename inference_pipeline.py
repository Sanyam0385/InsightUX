import numpy as np
import onnxruntime as ort
from scipy.interpolate import Rbf


# =============================================================================
# ONNX MODEL (FIXED FOR BINOCULAR v4)
# =============================================================================

class GazeONNXModel:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        # Detect input type automatically
        input_names = [inp.name for inp in self.session.get_inputs()]
        self.is_binocular = "eye_patch_binocular" in input_names

        print(f"ONNX loaded: {onnx_path}")
        print(f"Mode: {'binocular (v4)' if self.is_binocular else 'monocular'}")

        # Warmup
        if self.is_binocular:
            dummy_patch = np.zeros((1, 2, 36, 60), dtype=np.float32)
            dummy_pose  = np.zeros((1, 3), dtype=np.float32)
            self.session.run(
                ["gaze"],
                {"eye_patch_binocular": dummy_patch, "head_pose": dummy_pose}
            )
        else:
            dummy_patch = np.zeros((1, 1, 36, 60), dtype=np.float32)
            dummy_pose  = np.zeros((1, 3), dtype=np.float32)
            self.session.run(
                ["gaze"],
                {"eye_patch": dummy_patch, "head_pose": dummy_pose}
            )

    def predict(self, left_patch, right_patch, head_pose):
        def norm(img):
            return (img.astype(np.float32) / 255.0 - 0.5) / 0.5

        pose = head_pose.astype(np.float32)[np.newaxis, :]

        if self.is_binocular:
            patch = np.stack([norm(left_patch), norm(right_patch)], axis=0)
            patch = patch[np.newaxis, :, :, :]

            out = self.session.run(
                ["gaze"],
                {"eye_patch_binocular": patch, "head_pose": pose}
            )[0][0]

        else:
            patch = norm(left_patch)[np.newaxis, np.newaxis, :, :]

            out = self.session.run(
                ["gaze"],
                {"eye_patch": patch, "head_pose": pose}
            )[0][0]

        return float(out[0]), float(out[1])  # pitch, yaw


# =============================================================================
# RBF CALIBRATION (with extrapolation clamp)
# =============================================================================

class RBFGazeCalibration:
    def __init__(self, screen_w=1493, screen_h=933,
                 function="multiquadric", smooth=0.1):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.function = function
        self.smooth   = smooth
        self._rbf_x   = None
        self._rbf_y   = None
        self._calibrated = False

        # Range of (pitch, yaw) actually seen during calibration. Multiquadric
        # RBF surfaces are only well-behaved INSIDE the convex hull of their
        # training points. Outside it, the surface is dominated by whichever
        # basis functions happen to be nearest, and the gradient direction is
        # not guaranteed to continue the trend - it can curve back the WRONG
        # way. That's the textbook cause of "looks right, tracker says left"
        # specifically at extreme gaze angles, while mid-range tracking stays
        # correctly oriented. Clamping pitch/yaw to a small margin beyond the
        # calibrated range, before ever calling the RBF, keeps every query
        # inside (or just at the edge of) the well-behaved interpolation
        # region instead of letting it wander into extrapolation territory.
        self._pitch_range = None
        self._yaw_range   = None

    def calibrate(self, pitch_yaw, screen_points):
        pitch = pitch_yaw[:, 0]
        yaw   = pitch_yaw[:, 1]

        self._rbf_x = Rbf(pitch, yaw, screen_points[:, 0],
                          function=self.function, smooth=self.smooth)
        self._rbf_y = Rbf(pitch, yaw, screen_points[:, 1],
                          function=self.function, smooth=self.smooth)

        margin_p = 0.10 * (pitch.max() - pitch.min() + 1e-6)
        margin_y = 0.10 * (yaw.max()   - yaw.min()   + 1e-6)
        self._pitch_range = (float(pitch.min() - margin_p), float(pitch.max() + margin_p))
        self._yaw_range   = (float(yaw.min()   - margin_y), float(yaw.max()   + margin_y))

        self._calibrated = True
        print(f"RBF calibrated on {len(pitch_yaw)} points")
        print(f"  pitch clamp range: {self._pitch_range[0]:.4f} to {self._pitch_range[1]:.4f}")
        print(f"  yaw clamp range:   {self._yaw_range[0]:.4f} to {self._yaw_range[1]:.4f}")

    def predict(self, pitch, yaw):
        if not self._calibrated:
            raise RuntimeError("Calibration not done")

        # Clamp BEFORE calling the RBF, not after - clamping the screen
        # output instead would still let a reversed surface poison the
        # value before clamping catches it.
        if self._pitch_range is not None:
            pitch = float(np.clip(pitch, self._pitch_range[0], self._pitch_range[1]))
        if self._yaw_range is not None:
            yaw = float(np.clip(yaw, self._yaw_range[0], self._yaw_range[1]))

        sx = float(self._rbf_x(pitch, yaw))
        sy = float(self._rbf_y(pitch, yaw))

        sx = max(0, min(sx, self.screen_w))
        sy = max(0, min(sy, self.screen_h))

        return sx, sy

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "x": self._rbf_x,
                "y": self._rbf_y,
                "pitch_range": self._pitch_range,
                "yaw_range": self._yaw_range,
            }, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._rbf_x = data["x"]
        self._rbf_y = data["y"]
        # .get() so an OLDER calibration.pkl (saved before this fix) loads
        # fine too - it just runs without the clamp until you recalibrate.
        self._pitch_range = data.get("pitch_range")
        self._yaw_range   = data.get("yaw_range")
        self._calibrated = True


# =============================================================================
# FULL PIPELINE
# =============================================================================

class InsightUXPipeline:
    def __init__(self, onnx_path, calibration_path=None):
        self.gaze_model = GazeONNXModel(onnx_path)
        self.calibration = RBFGazeCalibration()

        if calibration_path:
            self.calibration.load(calibration_path)

    def predict_gaze_vector(self, left_patch, head_pose, right_patch=None):
        if right_patch is None:
            right_patch = left_patch

        pitch, yaw = self.gaze_model.predict(left_patch, right_patch, head_pose)

        gvx = -np.cos(pitch) * np.sin(yaw)
        gvy = -np.sin(pitch)

        return float(gvx), float(gvy), float(pitch), float(yaw)

    def predict_screen(self, left_patch, head_pose, right_patch=None):
        _, _, pitch, yaw = self.predict_gaze_vector(left_patch, head_pose, right_patch)
        return self.calibration.predict(pitch, yaw)