import cv2
import numpy as np
import time
import os
import random
import pyautogui

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
# FIX A — POSE NORMALIZATION
# solvePnP returns degrees (typically ±30 deg at a laptop, now that the
# FACE_3D_MODEL Y/Z sign bug in preprocessing_pipeline.py is fixed).
# The CNN stream_b was trained expecting values near [-1, +1].
# Dividing by 30 maps ±30 deg -> ±1, matching training scale.
# Keep POSE_NORM_SCALE identical in main_webcam_pipeline.py.
# =============================================================================
POSE_NORM_SCALE = 30.0

def normalize_pose(head_pose):
    return np.array([
        head_pose.pitch / POSE_NORM_SCALE,
        head_pose.yaw   / POSE_NORM_SCALE,
        head_pose.roll  / POSE_NORM_SCALE,
    ], dtype=np.float32)


# =============================================================================
# FIX B — HEAD-PITCH COMPENSATION
# The CNN partially learns absolute gaze = head_pitch + eye_rotation.
# When the head is still (laptop use) those are correlated and the model
# can't separate them. Subtracting a fraction of head pitch (converted
# to radians) from predicted pitch gives RBF real vertical signal.
# 0.4 = subtract 40 % of head pitch. Tune between 0.3-0.6 if needed:
#   still flat vertically -> raise toward 0.6
#   top/bottom flipped   -> lower toward 0.2
# Keep HEAD_PITCH_COMPENSATION identical in main_webcam_pipeline.py.
#
# NOTE: now that head_pose.pitch reports sane small values (the
# FACE_3D_MODEL fix), this term is a real, small correction instead of
# being swamped by a ~180deg garbage offset. Re-check whether 0.4 is
# still the right weight after recalibrating — it was tuned blind
# against the broken head pose, so it may need adjusting.
# =============================================================================
HEAD_PITCH_COMPENSATION = 0.0

def compensate_pitch(raw_pitch, head_pitch_deg):
    return raw_pitch - np.radians(head_pitch_deg) * HEAD_PITCH_COMPENSATION


# =============================================================================
# FIX C — HEAD-YAW COMPENSATION
# Same idea as FIX B, applied to the horizontal axis. If your head turns
# even slightly during a session (natural posture drift while reading),
# that motion reads as eye movement and pushes the prediction toward one
# edge - this is the likely cause of "accurate right after calibration,
# drifts toward one side a few seconds later." 0.3 is a starting point,
# tune between 0.2-0.5 the same way as HEAD_PITCH_COMPENSATION:
#   still drifting toward an edge  -> raise toward 0.5
#   left/right feels flipped/overcorrected -> lower toward 0.2
# Keep HEAD_YAW_COMPENSATION identical in calibrate.py / main_webcam_pipeline.py.
# =============================================================================
HEAD_YAW_COMPENSATION = 0.0  # disabled

def compensate_yaw(raw_yaw, head_yaw_deg):
    return raw_yaw - np.radians(head_yaw_deg) * HEAD_YAW_COMPENSATION


# =============================================================================
# PRE-CALIBRATION CHECKS
# =============================================================================

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 60:
        return False, "Too dark - increase lighting"
    elif brightness > 200:
        return False, "Too bright - reduce lighting"
    return True, None


def check_face_center(lms, frame_shape):
    H, W = frame_shape[:2]
    xs = [lm.x * W for lm in lms]
    ys = [lm.y * H for lm in lms]
    cx = np.mean(xs)
    cy = np.mean(ys)
    if cx < W * 0.35:
        return False, "Move face RIGHT in frame"
    elif cx > W * 0.65:
        return False, "Move face LEFT in frame"
    if cy < H * 0.35:
        return False, "Sit closer or lower your camera"
    elif cy > H * 0.65:
        return False, "Sit further or raise your camera"
    return True, None


def get_full_feedback(frame, lms, frame_shape, yaw, roll):
    msgs = []
    ok = True
    H, W = frame_shape[:2]

    nose_y     = lms[1].y   * H
    forehead_y = lms[10].y  * H
    chin_y     = lms[152].y * H
    midpoint_y = (forehead_y + chin_y) / 2.0
    face_h     = chin_y - forehead_y

    if face_h > 1:
        pitch_ratio = (nose_y - midpoint_y) / face_h
        if pitch_ratio < -0.10:
            msgs.append("Lift your head UP")
            ok = False
        elif pitch_ratio > 0.15:
            msgs.append("Tilt your head DOWN slightly")
            ok = False

    if yaw < -15:
        msgs.append("Turn face slightly RIGHT")
        ok = False
    elif yaw > 15:
        msgs.append("Turn face slightly LEFT")
        ok = False

    if roll < -10:
        msgs.append("Tilt head slightly RIGHT")
        ok = False
    elif roll > 10:
        msgs.append("Tilt head slightly LEFT")
        ok = False

    light_ok, light_msg = check_lighting(frame)
    if not light_ok:
        msgs.append(light_msg)
        ok = False

    center_ok, center_msg = check_face_center(lms, frame_shape)
    if not center_ok:
        msgs.append(center_msg)
        ok = False

    if ok:
        msgs.append("Perfect! Hold still...")

    return ok, msgs


# =============================================================================
# FACE ORIENTATION GATE
# =============================================================================

def face_orientation_gate(face_mesh, cap, cam_matrix_ref):
    print("\nChecking face setup before calibration...")
    print("Position your face straight, centred, at normal laptop distance.")

    HOLD_SECONDS   = 2.0
    good_since     = None
    cam_matrix_loc = cam_matrix_ref[0]

    cv2.namedWindow("Face Check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Check", 640, 420)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if cam_matrix_loc is None:
            cam_matrix_loc    = estimate_camera_matrix(frame.shape)
            cam_matrix_ref[0] = cam_matrix_loc

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        display = frame.copy()
        H, W    = display.shape[:2]

        if not results.multi_face_landmarks:
            good_since = None
            cv2.putText(display, "No face detected - look at camera",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            lms       = results.multi_face_landmarks[0].landmark
            head_pose = estimate_head_pose(lms, frame.shape, cam_matrix_loc)

            if head_pose is None:
                good_since = None
                cv2.putText(display, "Pose failed - move slightly",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                is_good, msgs = get_full_feedback(
                    frame, lms, frame.shape, head_pose.yaw, head_pose.roll
                )

                if is_good:
                    if good_since is None:
                        good_since = time.time()
                    elapsed   = time.time() - good_since
                    remaining = max(0, HOLD_SECONDS - elapsed)

                    bar_w = int((elapsed / HOLD_SECONDS) * (W - 40))
                    bar_w = min(bar_w, W - 40)
                    cv2.rectangle(display, (20, H-50), (W-20, H-25), (40, 40, 40), -1)
                    cv2.rectangle(display, (20, H-50), (20+bar_w, H-25), (0, 220, 0), -1)
                    cv2.putText(display, f"Hold still... {remaining:.1f}s",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if elapsed >= HOLD_SECONDS:
                        cv2.putText(display, "Starting calibration!",
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.imshow("Face Check", display)
                        cv2.waitKey(800)
                        cv2.destroyWindow("Face Check")
                        return
                else:
                    good_since = None
                    for i, msg in enumerate(msgs):
                        cv2.putText(display, msg, (20, 40 + i * 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

                cv2.putText(display,
                            f"Yaw:{head_pose.yaw:+.1f}  Roll:{head_pose.roll:+.1f}",
                            (20, H - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        cv2.imshow("Face Check", display)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyWindow("Face Check")
            return


# =============================================================================
# FINE-TUNING TARGET ANGLES
# nx/ny normalized to [-1,+1] so targets span real gaze range.
# =============================================================================

def screen_to_gaze_angles(sx, sy, screen_w, screen_h):
    nx    = (sx - screen_w  / 2) / (screen_w  / 2)
    ny    = (sy - screen_h / 2) / (screen_h / 2)
    pitch = float(np.arctan(ny * 0.4))
    yaw   = float(np.arctan(nx * 0.6))
    return pitch, yaw


def finetune_on_calibration_v4(
    onnx_path,
    left_patches, right_patches, head_poses, screen_points,
    screen_w, screen_h,
    ckpt_path="checkpoints/best_model_v4.pt",
    out_onnx_path=None,
    steps=300, lr=5e-4,
):
    try:
        import torch
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.model_v4 import GazeCNNv4
    except ImportError as e:
        print(f"[Fine-tune] Skipped: {e}")
        return False

    if not os.path.exists(ckpt_path):
        print(f"[Fine-tune] Skipped: checkpoint not found at {ckpt_path}")
        return False

    if out_onnx_path is None:
        out_onnx_path = onnx_path

    device = torch.device("cpu")
    model  = GazeCNNv4().to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # FIX: only fine-tune the small fusion head (fc_a + stream_b + fusion).
    # The original code also unfroze model.backbone[-2:], the last two
    # EfficientNet-B0 blocks - hundreds of thousands of extra weights -
    # and trained all of it (1.5M+ trainable params total) on ~65 frames.
    # That is a severe overfit risk: this exact danger (unfreezing backbone
    # blocks on a tiny calibration set) was flagged and rejected earlier in
    # this project. Backbone stays fully frozen here; only the lightweight
    # heads adapt to your eyes.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc_a.parameters():     p.requires_grad = True
    for p in model.stream_b.parameters(): p.requires_grad = True
    for p in model.fusion.parameters():   p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Fine-tune] Trainable: {trainable:,} parameters")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    def angular_loss(pred, target):
        def to_vec(a):
            p, y = a[:, 0], a[:, 1]
            v = torch.stack([torch.cos(p)*torch.sin(y),
                             torch.sin(p),
                             torch.cos(p)*torch.cos(y)], dim=1)
            return v / (v.norm(dim=1, keepdim=True) + 1e-8)
        cos_sim = (to_vec(pred) * to_vec(target)).sum(dim=1)
        return (1 - cos_sim).mean()

    patches_l, poses_t, gazes = [], [], []
    for l, r, pose, sp in zip(left_patches, right_patches, head_poses, screen_points):
        def norm(x): return (x.astype(np.float32) / 255.0 - 0.5) / 0.5
        patches_l.append(np.stack([norm(l), norm(r)], axis=0))
        poses_t.append(pose)   # already normalized (FIX A applied upstream)
        gazes.append(screen_to_gaze_angles(sp[0], sp[1], screen_w, screen_h))

    patches_t = torch.tensor(np.array(patches_l), dtype=torch.float32)
    poses_t   = torch.tensor(np.array(poses_t),   dtype=torch.float32)
    gazes_t   = torch.tensor(np.array(gazes),     dtype=torch.float32)

    # SPEED FIX: the backbone is fully frozen and the 75 input patches never
    # change across the 200 steps, so re-running the full EfficientNet-B0
    # forward pass every single step recomputes the exact same activations
    # 200 times for nothing - on CPU that's minutes of dead time before the
    # first "Step 50/200" print, which looks exactly like a hang.
    # Precompute the frozen backbone features ONCE, then iterate only over
    # the small trainable head (fc_a + stream_b + fusion). As a side effect
    # this also keeps the backbone's BatchNorm layers in eval mode (using
    # running stats) instead of accidentally switching them to batch
    # statistics over just 75 calibration frames, matching how the model
    # actually runs at inference time.
    model.eval()
    with torch.no_grad():
        feat_a_raw = model.pool(model.backbone(patches_t)).flatten(1)   # (N, 1280), frozen

    model.fc_a.train()
    model.stream_b.train()
    model.fusion.train()

    print(f"[Fine-tune] {steps} steps on {len(patches_l)} samples (backbone cached)...")
    for step in range(steps):
        optimizer.zero_grad()
        feat_a = model.fc_a(feat_a_raw)
        feat_b = model.stream_b(poses_t)
        fused  = torch.cat([feat_a, feat_b], dim=1)
        pred   = model.fusion(fused)
        loss   = angular_loss(pred, gazes_t)
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{steps} | Loss: {loss.item():.5f}")

    model.eval()
    dp    = torch.zeros(1, 2, 36, 60)
    dpose = torch.zeros(1, 3)
    torch.onnx.export(
        model, (dp, dpose), out_onnx_path,
        input_names  = ["eye_patch_binocular", "head_pose"],
        output_names = ["gaze"],
        dynamic_axes = {"eye_patch_binocular": {0: "batch"},
                        "head_pose": {0: "batch"}, "gaze": {0: "batch"}},
        opset_version=14,
        dynamo=False,
    )
    print(f"[Fine-tune] Adapted ONNX saved: {out_onnx_path}")
    return True


# =============================================================================
# OUTLIER FILTERING
# =============================================================================

def filter_outlier_points(gaze_vectors, screen_points, lefts, rights, poses,
                          head_pitches_deg, sigma=2.0):
    """
    Remove calibration points whose averaged gaze angle is >sigma std-devs
    from the median. Operates at the POINT level (not the expanded frame level)
    so lengths stay consistent across all parallel lists.
    Returns filtered versions of all six lists.
    """
    gv    = np.array(gaze_vectors)
    med_p = np.median(gv[:, 0])
    med_y = np.median(gv[:, 1])
    std_p = max(np.std(gv[:, 0]), 1e-6)
    std_y = max(np.std(gv[:, 1]), 1e-6)

    keep  = (np.abs(gv[:, 0] - med_p) / std_p <= sigma)
    keep &= (np.abs(gv[:, 1] - med_y) / std_y <= sigma)

    n_removed = int((~keep).sum())
    if n_removed:
        print(f"[Outlier filter] Removed {n_removed} point(s) (>{sigma}σ)")

    idx = np.where(keep)[0].tolist()
    return (
        [gaze_vectors[i]    for i in idx],
        [screen_points[i]   for i in idx],
        [lefts[i]           for i in idx],
        [rights[i]          for i in idx],
        [poses[i]           for i in idx],
        [head_pitches_deg[i] for i in idx],
    )


# =============================================================================
# CONFIG
# =============================================================================

ONNX_PATH = "models/gaze_cnn_v4.onnx"
CKPT_PATH = "checkpoints/best_model_v4.pt"

SCREEN_W, SCREEN_H = pyautogui.size()

points = [
    # Pushed to near the TRUE screen edges (0.04/0.96), not 0.1/0.9.
    # The old grid left a wide unsampled band at every edge, so any gaze
    # near an edge (navbar, page margins, the extremes you tested) was
    # RBF extrapolation, not interpolation - the likely cause of the
    # left/right reversal you saw when looking far to one side.
    (0.04, 0.04), (0.35, 0.04), (0.65, 0.04), (0.96, 0.04),
    (0.04, 0.35), (0.35, 0.35), (0.65, 0.35), (0.96, 0.35),
    (0.04, 0.65), (0.35, 0.65), (0.65, 0.65), (0.96, 0.65),
    (0.04, 0.96), (0.35, 0.96), (0.65, 0.96), (0.96, 0.96),
]

def get_duration(py):
    return 5 if py <= 0.4 else 3

# QUALITY FIX 1: shuffle point order before each session. Points were
# collected top-row-first, which means screen ROW and TIME ELAPSED were
# the same thing - any natural posture drift during the session (head
# settling, leaning back) showed up looking exactly like a real pitch
# trend by row, when some of it was just drift. Shuffling breaks that
# correlation: drift still happens, but now it's spread evenly across
# the whole grid instead of stacking up as a fake trend along one axis.
# Fixed seed so a session is reproducible if you need to debug it.
random.Random(7).shuffle(points)

FINETUNE_SAMPLES_PER_POINT = 5


# =============================================================================
# MAIN CALIBRATION
# =============================================================================

pipeline  = InsightUXPipeline(ONNX_PATH)
face_mesh = create_face_mesh(static_image_mode=False)
cap       = cv2.VideoCapture(0)

cam_matrix = None

# --- Per-point lists (one entry per calibration dot) ---
gaze_vectors     = []   # averaged (pitch, yaw) after FIX B compensation
screen_points    = []   # (sx, sy) pixel positions
head_pitches_deg = []   # average head pitch in degrees for this point
# One representative patch per point (used for RBF re-prediction after fine-tune)
rep_lefts  = []
rep_rights = []
rep_poses  = []
rep_hp_deg = []

# --- Expanded lists (FINETUNE_SAMPLES_PER_POINT entries per point) ---
ft_lefts  = []
ft_rights = []
ft_poses  = []
ft_sp     = []   # screen point duplicated to match expanded length

# Face orientation gate
cam_matrix_ref = [None]
face_orientation_gate(face_mesh, cap, cam_matrix_ref)
if cam_matrix_ref[0] is not None:
    cam_matrix = cam_matrix_ref[0]

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

total = len(points)
print(f"Calibration started - {total} points")

for idx, (px, py) in enumerate(points):
    sx, sy   = int(px * SCREEN_W), int(py * SCREEN_H)
    duration = get_duration(py)

    # QUALITY FIX 2: settle delay before EVERY point, not just the first.
    # Previously only point 1 had a pause before recording started, so
    # every later point began recording the instant the dot jumped there -
    # capturing the saccade itself (eyes still mid-flight, landmark
    # tracking least reliable) as real calibration data. A short settle
    # before each point discards that transition. Point 1 still gets a
    # longer pause since that's genuinely "get into position" time.
    settle_s = 1.5 if idx == 0 else 0.4
    settle_start = time.time()
    while time.time() - settle_start < settle_s:
        ret, frame = cap.read()
        if not ret:
            continue
        screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.circle(screen, (sx, sy), 18, (80, 80, 80), -1)
        msg = "Get ready..." if idx == 0 else "Settling..."
        cv2.putText(screen, msg, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)
        cv2.imshow("Calibration", screen)
        cv2.waitKey(1)

    samples    = []
    l_list     = []
    r_list     = []
    p_list     = []
    hp_list    = []   # head pitch in degrees per frame
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        if cam_matrix is None:
            cam_matrix = estimate_camera_matrix(frame.shape)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        lms       = results.multi_face_landmarks[0].landmark
        head_pose = estimate_head_pose(lms, frame.shape, cam_matrix)
        if head_pose is None:
            continue

        # FIX A: normalize pose to [-1, +1] before CNN
        pose_vec = normalize_pose(head_pose)

        def process_eye(eye_idx, ear_idx, iris_idx):
            s1 = step1_normalize(frame, lms, head_pose, eye_idx, ear_idx, iris_idx)
            if not s1.is_open:
                return None
            ir = compute_iris_radius(lms, iris_idx, frame.shape)
            s2 = step2_illumination(s1, ir)
            return s2.blended if s2.is_usable else None

        l = process_eye(LEFT_EYE_INDICES,  LEFT_EAR_INDICES,  LEFT_IRIS_INDICES)
        r = process_eye(RIGHT_EYE_INDICES, RIGHT_EAR_INDICES, RIGHT_IRIS_INDICES)

        if l is None and r is None:
            continue
        if l is None: l = r
        if r is None: r = l

        _, _, raw_pitch, raw_yaw = pipeline.predict_gaze_vector(l, pose_vec, r)

        # FIX B: remove head-pitch contribution from predicted pitch
        pitch = compensate_pitch(raw_pitch, head_pose.pitch)
        yaw   = compensate_yaw(raw_yaw, head_pose.yaw)

        samples.append([pitch, yaw])
        l_list.append(l)
        r_list.append(r)
        p_list.append(pose_vec)
        hp_list.append(head_pose.pitch)

        elapsed = time.time() - start
        screen  = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        angle   = int(360 * elapsed / duration)
        cv2.ellipse(screen, (sx, sy), (28, 28), -90, 0, angle, (0, 180, 0), 3)
        cv2.circle(screen, (sx, sy), 18, (0, 255, 0), -1)
        cv2.putText(screen, f"Point {idx+1}/{total} - Look at the dot",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Calibration", screen)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if not samples:
        print(f"Point {idx+1}: no samples, skipping.")
        continue

    # QUALITY FIX 3: median instead of mean. A single bad frame (blink
    # tail, momentary landmark jitter) can pull a mean noticeably off;
    # median ignores it as long as most frames in the window are good.
    avg        = np.median(samples, axis=0)
    avg_hp_deg = float(np.median(hp_list))

    # --- point-level lists ---
    gaze_vectors.append(list(avg))
    screen_points.append([sx, sy])
    head_pitches_deg.append(avg_hp_deg)

    # Representative frame (middle of recording window)
    mid = len(l_list) // 2
    rep_lefts.append(l_list[mid])
    rep_rights.append(r_list[mid])
    rep_poses.append(p_list[mid])
    rep_hp_deg.append(hp_list[mid])

    # --- expanded fine-tune lists ---
    n_avail = len(l_list)
    n_take  = min(FINETUNE_SAMPLES_PER_POINT, n_avail)
    indices = np.linspace(0, n_avail - 1, n_take, dtype=int)
    for i in indices:
        ft_lefts.append(l_list[i])
        ft_rights.append(r_list[i])
        ft_poses.append(p_list[i])
        ft_sp.append([sx, sy])

    print(f"Point {idx+1:2d}/{total} | pitch={avg[0]:.4f} yaw={avg[1]:.4f} "
          f"| head_pitch={avg_hp_deg:.1f}deg | {len(indices)} frames kept")

cap.release()
cv2.destroyAllWindows()

if len(gaze_vectors) < 4:
    print(f"ERROR: only {len(gaze_vectors)} points collected, need at least 4.")
    exit(1)

# =============================================================================
# OUTLIER FILTER  — operates on point-level lists only
# =============================================================================
(gaze_vectors, screen_points,
 rep_lefts, rep_rights, rep_poses,
 rep_hp_deg) = filter_outlier_points(
    gaze_vectors, screen_points,
    rep_lefts, rep_rights, rep_poses,
    head_pitches_deg, sigma=3.0
)

# Rebuild expanded fine-tune lists to match surviving points.
n_pts = len(screen_points)
if n_pts < 4:
    print("ERROR: too many outliers removed, need at least 4 clean points.")
    exit(1)

# Rebuild ft_* from the surviving rep frames (duplicated for fine-tuning)
ft_lefts_clean  = []
ft_rights_clean = []
ft_poses_clean  = []
ft_sp_clean     = []
for i in range(n_pts):
    for _ in range(FINETUNE_SAMPLES_PER_POINT):
        ft_lefts_clean.append(rep_lefts[i])
        ft_rights_clean.append(rep_rights[i])
        ft_poses_clean.append(rep_poses[i])
        ft_sp_clean.append(screen_points[i])

# =============================================================================
# FINE-TUNE
# =============================================================================
print(f"\n--- Fine-tuning CNN on your eyes ({len(ft_lefts_clean)} frames) ---")
finetuned = finetune_on_calibration_v4(
    onnx_path     = ONNX_PATH,
    left_patches  = ft_lefts_clean,
    right_patches = ft_rights_clean,
    head_poses    = ft_poses_clean,
    screen_points = ft_sp_clean,
    screen_w      = SCREEN_W,
    screen_h      = SCREEN_H,
    ckpt_path     = CKPT_PATH,
    out_onnx_path = ONNX_PATH,
    steps         = 80,
    lr            = 1e-4,
)

# =============================================================================
# FIT RBF CALIBRATION
# Re-predict gaze using the fine-tuned ONNX, apply FIX B compensation,
# then fit RBF. Uses exactly one rep frame per surviving point so
# len(adapted_gaze_vectors) == len(screen_points) always.
# =============================================================================
print("\n--- Fitting RBF calibration ---")
pipeline = InsightUXPipeline(ONNX_PATH)

adapted_gaze_vectors = []
for i in range(n_pts):
    l    = rep_lefts[i]
    r    = rep_rights[i]
    pose = rep_poses[i]   # already normalized (FIX A): [pitch, yaw, roll] / 30
    hp   = rep_hp_deg[i]
    hy   = float(pose[1]) * POSE_NORM_SCALE   # recover raw head yaw degrees

    _, _, raw_pitch, raw_yaw = pipeline.predict_gaze_vector(l, pose, r)

    # FIX B / FIX C: same compensation as during collection
    pitch = compensate_pitch(raw_pitch, hp)
    yaw   = compensate_yaw(raw_yaw, hy)
    adapted_gaze_vectors.append([pitch, yaw])

# Lengths guaranteed equal here
assert len(adapted_gaze_vectors) == len(screen_points), \
    f"Length mismatch: {len(adapted_gaze_vectors)} gaze vs {len(screen_points)} screen"

pipeline.calibration.calibrate(
    np.array(adapted_gaze_vectors),
    np.array(screen_points)
)
pipeline.calibration.save("calibration.pkl")

# Save baseline head pose separately for diagnostics/live scripts.
# Compensation is currently OFF, so this does not change prediction.
import pickle
baseline_pitch = float(np.median(rep_hp_deg)) if len(rep_hp_deg) else 0.0
baseline_yaw = float(np.median([float(p[1]) * POSE_NORM_SCALE for p in rep_poses])) if len(rep_poses) else 0.0
baseline_roll = float(np.median([float(p[2]) * POSE_NORM_SCALE for p in rep_poses])) if len(rep_poses) else 0.0
with open("baseline_pose.pkl", "wb") as f:
    pickle.dump({"pitch": baseline_pitch, "yaw": baseline_yaw, "roll": baseline_roll}, f)
print(f"Saved baseline_pose.pkl: pitch={baseline_pitch:+.2f}, yaw={baseline_yaw:+.2f}, roll={baseline_roll:+.2f}")

print(f"\nCalibration complete! {n_pts}/{total} points used.")
if finetuned:
    print("CNN fine-tuned on your eyes + RBF calibration fitted.")
else:
    print("RBF calibration fitted (CNN fine-tuning skipped).")

# --- Diagnostic: verify vertical pitch separation ---
print("\n--- Pitch by screen row (should increase top -> bottom) ---")
gv_arr = np.array(adapted_gaze_vectors)
sp_arr = np.array(screen_points)
thresholds = [(0,          SCREEN_H*0.25, "Top    (y<25%)  "),
              (SCREEN_H*0.25, SCREEN_H*0.5,  "Mid-hi (25-50%) "),
              (SCREEN_H*0.5,  SCREEN_H*0.75, "Mid-lo (50-75%) "),
              (SCREEN_H*0.75, SCREEN_H+1,    "Bottom (y>75%)  ")]
for lo, hi, label in thresholds:
    mask = (sp_arr[:, 1] >= lo) & (sp_arr[:, 1] < hi)
    if mask.any():
        print(f"  {label}: avg pitch = {gv_arr[mask, 0].mean():.4f}")

print("\n--- Yaw by screen column (should increase left -> right) ---")
col_thresholds = [(0,            SCREEN_W*0.25, "Left   (x<25%)  "),
                   (SCREEN_W*0.25, SCREEN_W*0.5,  "Mid-lf (25-50%) "),
                   (SCREEN_W*0.5,  SCREEN_W*0.75, "Mid-rt (50-75%) "),
                   (SCREEN_W*0.75, SCREEN_W+1,    "Right  (x>75%)  ")]
for lo, hi, label in col_thresholds:
    mask = (sp_arr[:, 0] >= lo) & (sp_arr[:, 0] < hi)
    if mask.any():
        print(f"  {label}: avg yaw = {gv_arr[mask, 1].mean():.4f}")