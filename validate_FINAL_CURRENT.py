"""
validate.py
Measures how accurate the tracker is after calibration, with real numbers.

Run AFTER calibrate.py (it loads calibration.pkl).
    python validate.py

It flashes 9 targets at positions BETWEEN your calibration points (so this is a
fair generalization test, not the dots the RBF was fit on). For each target it
collects gaze for a couple of seconds, takes the median predicted screen point,
and compares to the true target.

Prints:
    - mean pixel error
    - mean error as a percent of screen diagonal
    - zone hit rate: did the gaze land in the correct third of the screen
      (a 3x3 grid). This is the closest proxy to your AOI hit rate.

PATCH_SOURCE must match calibrate.py and run_session.py.
"""

import cv2
import numpy as np
import time
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


ONNX_PATH        = "models/gaze_cnn_v4.onnx"
CALIBRATION_PATH = "calibration.pkl"

SCREEN_W, SCREEN_H = pyautogui.size()
PATCH_SOURCE     = "blended"   # MUST match calibrate.py and main_webcam_pipeline.py

# FIX A / FIX B — MUST match calibrate.py and main_webcam_pipeline.py exactly,
# or this validation measures a different pipeline than the one calibrated.
POSE_NORM_SCALE         = 30.0
HEAD_PITCH_COMPENSATION = 0.0

def normalize_pose(head_pose):
    return np.array([
        head_pose.pitch / POSE_NORM_SCALE,
        head_pose.yaw   / POSE_NORM_SCALE,
        head_pose.roll  / POSE_NORM_SCALE,
    ], dtype=np.float32)

def compensate_pitch(raw_pitch, head_pitch_deg):
    return raw_pitch - np.radians(head_pitch_deg) * HEAD_PITCH_COMPENSATION

# Test targets between the calibration grid (fair generalization test)
TEST_POINTS = [
    (0.25, 0.25), (0.50, 0.25), (0.75, 0.25),
    (0.25, 0.50), (0.50, 0.50), (0.75, 0.50),
    (0.25, 0.75), (0.50, 0.75), (0.75, 0.75),
]
DURATION = 2.5   # seconds collected per target


def zone(sx, sy):
    col = 0 if sx < SCREEN_W / 3 else (1 if sx < 2 * SCREEN_W / 3 else 2)
    row = 0 if sy < SCREEN_H / 3 else (1 if sy < 2 * SCREEN_H / 3 else 2)
    return row, col


def get_patch(frame, lms, head_pose, eye_idx, ear_idx, iris_idx):
    s1 = step1_normalize(frame, lms, head_pose, eye_idx, ear_idx, iris_idx)
    if not s1.is_open:
        return None
    if PATCH_SOURCE == "norm":
        return s1.norm_crop
    ir = compute_iris_radius(lms, iris_idx, frame.shape)
    s2 = step2_illumination(s1, ir)
    return s2.blended if s2.is_usable else None


def main():
    pipeline  = InsightUXPipeline(ONNX_PATH, CALIBRATION_PATH)
    face_mesh = create_face_mesh(static_image_mode=False)
    cap       = cv2.VideoCapture(0)
    cam_matrix = None

    cv2.namedWindow("Validate", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Validate", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Validation: look at each red dot until it turns green.")
    print("Cyan dot = your live, single-frame prediction (will jitter, that's normal).")
    print("Magenta ring = the running median - this is what actually gets scored.")
    results = []   # (true_x, true_y, pred_x, pred_y)

    for idx, (px, py) in enumerate(TEST_POINTS):
        tx, ty = int(px * SCREEN_W), int(py * SCREEN_H)
        preds     = []
        last_pred = None
        start     = time.time()

        while time.time() - start < DURATION:
            ret, frame = cap.read()
            if not ret:
                continue
            if cam_matrix is None:
                cam_matrix = estimate_camera_matrix(frame.shape)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res     = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lms       = res.multi_face_landmarks[0].landmark
                head_pose = estimate_head_pose(lms, frame.shape, cam_matrix)
                if head_pose is not None:
                    pose_vec = normalize_pose(head_pose)   # FIX A
                    lp = get_patch(frame, lms, head_pose,
                                   LEFT_EYE_INDICES, LEFT_EAR_INDICES, LEFT_IRIS_INDICES)
                    rp = get_patch(frame, lms, head_pose,
                                   RIGHT_EYE_INDICES, RIGHT_EAR_INDICES, RIGHT_IRIS_INDICES)
                    if lp is not None or rp is not None:
                        if lp is None: lp = rp
                        if rp is None: rp = lp
                        _, _, raw_pitch, raw_yaw = pipeline.predict_gaze_vector(lp, pose_vec, rp)
                        pitch = compensate_pitch(raw_pitch, head_pose.pitch)   # FIX B
                        sx, sy = pipeline.calibration.predict(pitch, raw_yaw)
                        sx = max(0.0, min(sx, SCREEN_W))
                        sy = max(0.0, min(sy, SCREEN_H))
                        preds.append([sx, sy])
                        last_pred = (sx, sy)

            screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            ready  = len(preds) > 10
            color  = (0, 255, 0) if ready else (0, 0, 255)
            cv2.circle(screen, (tx, ty), 20, color, -1)
            cv2.putText(screen, f"Target {idx+1}/{len(TEST_POINTS)}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # LIVE single-frame prediction - this is exactly where the model
            # thinks you're looking RIGHT NOW. It will jitter frame to frame,
            # that's expected and not itself a problem.
            if last_pred is not None:
                lx, ly = int(last_pred[0]), int(last_pred[1])
                cv2.line(screen, (tx, ty), (lx, ly), (120, 120, 0), 1)
                cv2.circle(screen, (lx, ly), 9, (255, 255, 0), -1)

            # Running median across this target's samples so far - THIS is
            # the number that actually gets scored at the end, not the raw
            # jittery dot above. Watching it should settle near the red/
            # green dot as samples accumulate, if it settles somewhere else
            # entirely, that's a real miscalibration, not noise.
            if len(preds) >= 5:
                mx_, my_ = np.median(np.array(preds), axis=0)
                mxi, myi = int(mx_), int(my_)
                cv2.circle(screen, (mxi, myi), 16, (255, 0, 255), 2)
                live_err = float(np.hypot(tx - mx_, ty - my_))
                cv2.putText(screen, f"running error: {live_err:.0f}px",
                            (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(screen, "cyan = live   magenta ring = running median (scored)",
                        (50, SCREEN_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)

            cv2.imshow("Validate", screen)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release(); cv2.destroyAllWindows(); return

        if len(preds) >= 5:
            mx, my = np.median(np.array(preds), axis=0)
            results.append((tx, ty, float(mx), float(my)))
            err = float(np.hypot(tx - mx, ty - my))
            print(f"Target {idx+1}: true=({tx},{ty}) pred=({mx:.0f},{my:.0f})  error={err:.0f}px")

            # freeze-frame: show the final result for a beat before advancing,
            # so you can actually see how close it landed instead of it
            # flashing straight to the next target
            freeze = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(freeze, (tx, ty), 20, (0, 255, 0), -1)
            cv2.circle(freeze, (int(mx), int(my)), 16, (255, 0, 255), 2)
            cv2.line(freeze, (tx, ty), (int(mx), int(my)), (255, 0, 255), 2)
            cv2.putText(freeze, f"Target {idx+1}/{len(TEST_POINTS)}  error: {err:.0f}px",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Validate", freeze)
            cv2.waitKey(700)
        else:
            print(f"Target {idx+1}: too few samples, skipped")

    cap.release()
    cv2.destroyAllWindows()

    if not results:
        print("No valid targets. Check lighting and camera.")
        return

    errs = [np.hypot(tx - mx, ty - my) for (tx, ty, mx, my) in results]
    diag = np.hypot(SCREEN_W, SCREEN_H)
    hits = sum(1 for (tx, ty, mx, my) in results if zone(tx, ty) == zone(mx, my))

    print("\n================ VALIDATION RESULT ================")
    print(f"Targets measured     : {len(results)}/{len(TEST_POINTS)}")
    print(f"Mean pixel error     : {np.mean(errs):.0f} px")
    print(f"Median pixel error   : {np.median(errs):.0f} px")
    print(f"Mean error vs screen : {100*np.mean(errs)/diag:.1f}% of diagonal")
    print(f"Zone hit rate (3x3)  : {hits}/{len(results)}  ({100*hits/len(results):.0f}%)")
    print("===================================================")
    print("Zone hit rate is the closest proxy to AOI accuracy. Aim for a coarser")
    print("AOI layout than 3x3 if you need a higher number for the demo.")


if __name__ == "__main__":
    main()