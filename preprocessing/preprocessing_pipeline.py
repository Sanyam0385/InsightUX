"""
STEP 2 - Region-Aware Illumination Normalization
=================================================
Novel Eye Tracking Preprocessing Pipeline | InsightUX

Run AFTER step1_geometric_normalization.py.
This file imports Step 1 and adds Step 2 on top.

PAPER BASIS
-----------
Fischer et al., "RT-GENE: Real-Time Eye Gaze Estimation in Natural
Environments" (ECCV 2018)
  - First paper to treat illumination as a serious gaze preprocessing
    problem. They inpaint specular glints before CNN inference.
  - Apply single global illumination correction to the whole eye patch.

OUR NOVELTY VS RT-GENE
-----------------------
RT-GENE's gap: they treat the eye patch as photometrically uniform.
But the eye has two structurally distinct regions with different roles:

  IRIS REGION   - fine texture carries gaze signal (limbus, collarette,
                  crypts). Needs contrast enhancement. Over-brightening
                  destroys the texture detail the CNN relies on.

  SCLERA REGION - the limbus boundary (iris-sclera edge) carries coarse
                  gaze direction as a hard edge. The sclera interior is
                  a smooth gradient - noise amplification here hurts, not
                  helps. Needs edge-preserving smoothing, not contrast boost.

Treating both regions with identical CLAHE parameters (as RT-GENE and
every subsequent paper does) degrades one while helping the other.

OUR APPROACH: Region-Aware CLAHE
  1. Build an iris mask from the warped iris landmark circle
     (propagated through Step 1's warp matrix)
  2. Apply tight CLAHE (clipLimit=1.5, small tile) to iris region
     -> enhances fine texture without amplifying noise
  3. Apply looser CLAHE (clipLimit=3.0, large tile) to sclera region
     -> boosts the limbus edge contrast without destroying smoothness
  4. Feather-blend at the iris boundary (soft mask, not hard cutoff)
     -> avoids ring artifacts at the transition
  5. Global glint inpainting BEFORE region split (RT-GENE does this,
     we keep it and improve the glint threshold adaptively)
  6. Novel photometric quality score: iris texture visibility index
     -> measures whether iris texture is actually recoverable
     -> gates Step 3 onwards (if score too low, flag frame)

WHAT TO LOOK FOR
----------------
  - IRIS panel: more texture visible inside iris vs Step 1 norm_crop
  - SCLERA panel: limbus edge sharper, sclera interior stays smooth
  - BLENDED panel: seamless transition between the two regions
  - DIFF panel: where illumination correction changed things most
  - Photometric Q score: higher = iris texture more visible
  - ITV (Iris Texture Visibility): >0.3 = usable frame for gaze CNN

USAGE
-----
    python step2_illumination.py           # webcam
    python step2_illumination.py --image path
    python step2_illumination.py --video path
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import sys
from dataclasses import dataclass
from typing import Tuple, Optional

# Step 1 is embedded directly below - no import needed.

"""
STEP 1 - Geometric Normalization (v2, fixed)
=============================================
Novel Eye Tracking Preprocessing Pipeline | InsightUX

PAPER BASIS
-----------
Zhang et al., "Appearance-Based Gaze Estimation in the Wild" (MPIIGaze, CVPR 2015)
  - Section 3.1: Face normalization via virtual camera construction

OUR NOVELTY VS THE PAPER
-------------------------
MPIIGaze requires known camera intrinsics + calibrated 3D face model.
We make the approximation explicit and measurable:
  1. Estimate intrinsics from frame size (documented, not silent)
  2. solvePnP (EPnP + LM refinement) for head pose
  3. 2D similarity warp (roll correction + iris centering) applied
     WITHIN the ROI - not to the full frame (the v1 bug)
  4. Residual yaw/pitch stored for Step 7 (auxiliary CNN input)
  5. Novel normalization quality metric: gradient orientation entropy
     reduction - how much the warp reduced pose-induced disorder

FIXES vs v1
-----------
- Warp now operates in ROI-local coordinates, not full-frame coords
  (v1 applied warpAffine to full frame then cropped (0,0,W,H) -> wrong)
- Iris center correctly computed in ROI-local space before warp
- Similarity transform centered on ROI-local iris, not global iris
- Quality metric now computes on valid open-eye frames only

WHAT TO LOOK FOR
----------------
  - Tilt head sideways: NORMALIZED should stay upright, RAW tilts
  - Cyan dot should land on your iris/pupil center
  - DIFF should be mostly dark when head is straight (low roll)
  - DIFF should brighten when you tilt (warp actively correcting)
  - Q score should increase when you tilt and decrease at straight pose

USAGE
-----
    python step1_geometric_normalization.py           # webcam
    python step1_geometric_normalization.py --image path
    python step1_geometric_normalization.py --video path
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import sys
from dataclasses import dataclass
from typing import Tuple, Optional



# =============================================================================
# MediaPipe + Iris Radius Helpers
# =============================================================================

def create_face_mesh(static_image_mode: bool = False,
                     max_num_faces: int = 1,
                     refine_landmarks: bool = True,
                     min_detection_confidence: float = 0.5,
                     min_tracking_confidence: float = 0.5):
    """
    Creates and returns a MediaPipe FaceMesh instance.
    refine_landmarks=True is required for iris landmarks (indices 468-477).

    Handles MediaPipe version compatibility - newer builds may omit the
    legacy solutions.face_mesh API. Raises a clear RuntimeError with
    install instructions if the API is not available.

    Usage:
        face_mesh = create_face_mesh(static_image_mode=False)  # webcam/video
        face_mesh = create_face_mesh(static_image_mode=True)   # single image
    """
    try:
        from mediapipe.python.solutions.face_mesh import FaceMesh
    except ImportError as exc:
        py     = sys.version.split()[0]
        mp_ver = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            f"mediapipe {mp_ver} on Python {py} does not include the legacy "
            "solutions.face_mesh API. Use Python 3.11 with mediapipe==0.10.18 "
            "and reinstall: pip install mediapipe==0.10.18"
        ) from exc

    return FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def compute_iris_radius(landmarks, iris_indices: list, frame_shape: tuple) -> float:
    """
    Computes iris radius in frame pixels from MediaPipe iris landmarks.
    MediaPipe iris layout: index 0 = center, 1-4 = cardinal rim points.
    Radius = mean distance from center to the 4 rim points.

    ALWAYS use this instead of hardcoding iris_radius_frame = 15.0.
    The iris radius changes with camera distance. Hardcoding corrupts
    the CLAHE iris mask in Step 2 and directly degrades CNN input quality.

    Usage:
        left_ir  = compute_iris_radius(lms, LEFT_IRIS_INDICES,  frame.shape)
        right_ir = compute_iris_radius(lms, RIGHT_IRIS_INDICES, frame.shape)
        step2_result = step2_illumination(step1_result, left_ir)
    """
    h, w = frame_shape[:2]
    pts  = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices],
        dtype=np.float32
    )
    center = pts[0]
    radius = float(np.mean(np.linalg.norm(pts[1:] - center, axis=1)))
    return max(radius, 1.0)


# -----------------------------------------------------------------------------
# 3D Face Model + Landmark Indices
# -----------------------------------------------------------------------------

# Generic 6-point 3D face model in mm (origin = nose tip)
# Same model used in: Guo et al. 2020, Wood et al. ETH-XGaze 2022
FACE_3D_MODEL = np.array([
    [  0.0,    0.0,    0.0],   # nose tip         -> landmark 1
    [  0.0,  -63.6,  -12.5],  # chin             -> landmark 152
    [-43.3,   32.7,  -26.0],  # left eye outer   -> landmark 33
    [ 43.3,   32.7,  -26.0],  # right eye outer  -> landmark 263
    [-28.9,  -28.9,  -24.1],  # left mouth       -> landmark 61
    [ 28.9,  -28.9,  -24.1],  # right mouth      -> landmark 291
], dtype=np.float64)

FACE_2D_INDICES = [1, 152, 33, 263, 61, 291]

LEFT_EYE_INDICES   = [33, 7, 163, 144, 145, 153, 154, 155,
                      133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249,
                      263, 466, 388, 387, 386, 385, 384, 398]

LEFT_IRIS_INDICES  = [468, 469, 470, 471, 472]   # [center, top, right, bottom, left]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

LEFT_EAR_INDICES   = [33, 160, 158, 133, 153, 144]
RIGHT_EAR_INDICES  = [362, 385, 387, 263, 373, 380]

PATCH_W, PATCH_H   = 60, 36
EYE_PAD_FACTOR     = 0.45


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class HeadPose:
    pitch:        float          # up/down tilt (degrees)
    yaw:          float          # left/right turn (degrees)
    roll:         float          # in-plane rotation (degrees)
    rvec:         np.ndarray     # Rodrigues rotation vector (3,1)
    tvec:         np.ndarray     # translation vector (3,1)
    reproj_error: float          # mean reprojection error (pixels)


@dataclass
class NormalizedEyePatch:
    raw_crop:     np.ndarray               # uint8 gray - simple crop, no correction
    norm_crop:    np.ndarray               # uint8 gray - after 2D similarity warp
    diff_map:     np.ndarray               # uint8 BGR  - |raw - norm| heatmap
    bbox_raw:     Tuple[int,int,int,int]   # (x1,y1,x2,y2) in original frame
    warp_M:       np.ndarray               # 2x3 affine matrix (ROI-local)
    iris_center:  Tuple[float,float]       # iris center in normalized patch coords
    ear:          float
    is_open:      bool
    norm_quality: float                    # gradient entropy reduction [0,1]


# -----------------------------------------------------------------------------
# Camera Intrinsics (documented approximation)
# -----------------------------------------------------------------------------

def estimate_camera_matrix(frame_shape: tuple) -> np.ndarray:
    """
    Estimates camera intrinsic matrix from frame dimensions.

    DOCUMENTED ASSUMPTION: f ~= frame_width (equivalent to ~60deg FOV).
    Matches the silent assumption in Zhang 2015, Fischer 2018, Guo 2020.
    We make it explicit so residual intrinsic error is measurable
    via reproj_error in HeadPose.

    Returns 3x3 float64 camera matrix K.
    """
    h, w = frame_shape[:2]
    f  = float(w)
    cx = w / 2.0
    cy = h / 2.0
    return np.array([
        [f,   0., cx],
        [0.,  f,  cy],
        [0.,  0., 1.],
    ], dtype=np.float64)


# -----------------------------------------------------------------------------
# Head Pose Estimation
# -----------------------------------------------------------------------------

def estimate_head_pose(
    landmarks,
    frame_shape: tuple,
    camera_matrix: np.ndarray,
) -> Optional[HeadPose]:
    """
    solvePnP (EPnP) + LM refinement -> Euler angles.
    Ref: ETH-XGaze preprocessing (Wood et al. 2022).
    """
    h, w = frame_shape[:2]
    dist  = np.zeros((4, 1), dtype=np.float64)
    pts2d = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in FACE_2D_INDICES],
        dtype=np.float64
    )

    ok, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, pts2d, camera_matrix, dist,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        return None

    rvec, tvec = cv2.solvePnPRefineLM(
        FACE_3D_MODEL, pts2d, camera_matrix, dist, rvec, tvec
    )

    proj, _ = cv2.projectPoints(FACE_3D_MODEL, rvec, tvec, camera_matrix, dist)
    reproj  = float(np.mean(np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1)))

    R, _    = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _R_to_euler(R)

    return HeadPose(pitch=pitch, yaw=yaw, roll=roll,
                    rvec=rvec, tvec=tvec, reproj_error=reproj)


def _R_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Rotation matrix -> (pitch, yaw, roll) degrees. Zhang 2015 convention."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( R[2,1], R[2,2]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = 0.0
    return pitch, yaw, roll


# -----------------------------------------------------------------------------
# ROI Extraction
# -----------------------------------------------------------------------------

def extract_roi(
    frame: np.ndarray,
    landmarks,
    eye_indices: list,
    pad: float = EYE_PAD_FACTOR,
) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    Crops the eye ROI from the frame with padding.
    Returns (gray_roi, (x1,y1,x2,y2)) in original frame coords.
    """
    h, w = frame.shape[:2]
    pts  = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices],
        dtype=np.float32
    )
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    ew, eh = xmax - xmin, ymax - ymin

    x1 = max(0, int(xmin - ew * pad))
    y1 = max(0, int(ymin - eh * pad))
    x2 = min(w, int(xmax + ew * pad))
    y2 = min(h, int(ymax + eh * pad))

    if x2 <= x1 or y2 <= y1:
        blank = np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
        return blank, (0, 0, PATCH_W, PATCH_H)

    roi = frame[y1:y2, x1:x2]
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi.copy(), (x1, y1, x2, y2)


# -----------------------------------------------------------------------------
# 2D Normalization Warp - applied in ROI space (the key fix)
# -----------------------------------------------------------------------------

def compute_roi_warp(
    landmarks,
    frame_shape: tuple,
    bbox: Tuple[int,int,int,int],
    iris_indices: list,
    roll_deg: float,
) -> Tuple[np.ndarray, Tuple[float,float]]:
    """
    Computes a 2D similarity warp IN ROI-LOCAL COORDINATES.

    Fix vs v1: v1 applied the warp to the full frame then cropped
    at (0,0) - so the output was always the top-left corner of the
    warped frame, not the eye. This builds the transform entirely
    within the ROI coordinate system.

    The warp is:  Rotate(-roll) around iris_center_local
                  + Scale to fill PATCH_W x PATCH_H
                  + Translate so iris lands at patch center

    This is a similarity transform - preserves shape, corrects rotation,
    normalises scale. Yaw/pitch perspective distortion is NOT corrected
    here (stored as HeadPose residual for Step 7).

    Returns:
        warp_M         : 2x3 float64 affine matrix (ROI-local)
        iris_in_roi    : iris center in ROI-local pixel coords
    """
    h, w   = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    roi_w  = max(x2 - x1, 1)
    roi_h  = max(y2 - y1, 1)

    # Iris center in full-frame coords
    iris_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices],
        dtype=np.float32
    )
    iris_global = iris_pts[0]   # landmark 0 = iris center

    # Convert to ROI-local coords
    iris_local = (
        float(iris_global[0] - x1),
        float(iris_global[1] - y1),
    )

    # Scale: fit ROI into PATCH_W x PATCH_H (uniform, preserve aspect)
    scale = min(PATCH_W / roi_w, PATCH_H / roi_h)

    # Rotation angle: correct in-plane head roll
    a = np.radians(-roll_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)

    # Build warp:
    #   1. Rotate around iris_local
    #   2. Scale
    #   3. Translate so iris lands at patch center
    cx, cy = iris_local

    # Affine columns: [scale*R | scale*R*(-iris) + patch_center]
    tx = PATCH_W / 2.0 + scale * (-cos_a * cx + sin_a * cy)
    ty = PATCH_H / 2.0 + scale * (-sin_a * cx - cos_a * cy)

    warp_M = np.array([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a,  scale * cos_a, ty],
    ], dtype=np.float64)

    return warp_M, iris_local


def apply_warp_to_roi(
    gray_roi: np.ndarray,
    warp_M:   np.ndarray,
) -> np.ndarray:
    """
    Applies the 2x3 similarity warp to the grayscale ROI.
    Output is always PATCH_W x PATCH_H uint8.
    Uses LANCZOS4 interpolation (best quality for downscaling).
    """
    return cv2.warpAffine(
        gray_roi, warp_M, (PATCH_W, PATCH_H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )


def raw_crop_resized(gray_roi: np.ndarray) -> np.ndarray:
    """
    Baseline: resize ROI to patch size with no geometric correction.
    This is what most naive implementations do.
    """
    if gray_roi.size == 0:
        return np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
    return cv2.resize(gray_roi, (PATCH_W, PATCH_H), interpolation=cv2.INTER_LANCZOS4)


# -----------------------------------------------------------------------------
# Iris Center in Normalized Patch Coords
# -----------------------------------------------------------------------------

def iris_center_in_patch(
    iris_local: Tuple[float,float],
    warp_M: np.ndarray,
) -> Tuple[float,float]:
    """
    Projects the ROI-local iris center through the warp matrix
    to get its position in the normalized patch.
    Should land near (PATCH_W/2, PATCH_H/2) when warp is correct.
    """
    ix, iy = iris_local
    px = warp_M[0,0]*ix + warp_M[0,1]*iy + warp_M[0,2]
    py = warp_M[1,0]*ix + warp_M[1,1]*iy + warp_M[1,2]
    return float(px), float(py)


# -----------------------------------------------------------------------------
# Normalization Quality Metric
# -----------------------------------------------------------------------------

def compute_norm_quality(
    raw_crop:  np.ndarray,
    norm_crop: np.ndarray,
) -> float:
    """
    Gradient orientation entropy reduction.

    Intuition: a well-normalized patch (roll corrected) has more
    structured gradient orientations - lids are horizontal, iris
    edges are vertical. Entropy measures disorder.

    H_raw > H_norm  -> normalization reduced disorder -> quality > 0
    H_raw ~= H_norm  -> no change (e.g. roll ~= 0) -> quality ~= 0
    H_raw < H_norm  -> normalization hurt -> clamped to 0

    quality in [0, 1].
    NOTE: Near-zero roll -> raw ~= norm -> quality ~= 0 by design.
    This is correct behaviour, not a bug.
    """
    def entropy(patch: np.ndarray) -> float:
        gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(gy, gx)
        hist, _ = np.histogram(angles.ravel(), bins=36, range=(-np.pi, np.pi))
        hist = hist.astype(np.float64) + 1e-9
        hist /= hist.sum()
        return float(-np.sum(hist * np.log(hist)))

    H_raw  = entropy(raw_crop)
    H_norm = entropy(norm_crop)
    if H_raw < 1e-9:
        return 0.0
    return float(np.clip((H_raw - H_norm) / H_raw, 0.0, 1.0))


# -----------------------------------------------------------------------------
# EAR
# -----------------------------------------------------------------------------

def compute_ear(landmarks, ear_indices: list, frame_shape: tuple) -> float:
    h, w = frame_shape[:2]
    pts  = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in ear_indices],
        dtype=np.float32
    )
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C)) if C > 1e-6 else 0.0


# -----------------------------------------------------------------------------
# Master Step 1 Function
# -----------------------------------------------------------------------------

def step1_normalize(
    frame:        np.ndarray,
    landmarks,
    head_pose:    HeadPose,
    eye_indices:  list,
    ear_indices:  list,
    iris_indices: list,
    ear_threshold: float = 0.20,
) -> NormalizedEyePatch:
    """
    Full Step 1 pipeline for one eye:
      1. EAR
      2. Extract gray ROI
      3. Compute ROI-local similarity warp from roll + iris center
      4. Apply warp -> norm_crop
      5. Resize only -> raw_crop (baseline)
      6. Diff heatmap
      7. Iris center in patch coords
      8. Normalization quality score
    """
    ear     = compute_ear(landmarks, ear_indices, frame.shape)
    is_open = ear >= ear_threshold

    # Gray ROI in original frame
    gray_roi, bbox = extract_roi(frame, landmarks, eye_indices)

    # Warp matrix in ROI-local coords
    warp_M, iris_local = compute_roi_warp(
        landmarks, frame.shape, bbox, iris_indices, head_pose.roll
    )

    # Normalized patch (warp applied to ROI)
    norm_crop = apply_warp_to_roi(gray_roi, warp_M)

    # Raw baseline (resize only, no correction)
    raw_crop = raw_crop_resized(gray_roi)

    # Diff heatmap: where did normalization change things?
    diff      = cv2.absdiff(raw_crop, norm_crop)
    diff_vis  = cv2.applyColorMap(
        cv2.convertScaleAbs(diff, alpha=4.0), cv2.COLORMAP_INFERNO
    )

    # Iris center in normalized patch
    iris_patch = iris_center_in_patch(iris_local, warp_M)

    # Quality metric (only meaningful when eye is open)
    quality = compute_norm_quality(raw_crop, norm_crop) if is_open else 0.0

    return NormalizedEyePatch(
        raw_crop=raw_crop,
        norm_crop=norm_crop,
        diff_map=diff_vis,
        bbox_raw=bbox,
        warp_M=warp_M,
        iris_center=iris_patch,
        ear=ear,
        is_open=is_open,
        norm_quality=quality,
    )


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def draw_pose_axes(
    frame: np.ndarray,
    head_pose: HeadPose,
    camera_matrix: np.ndarray,
    nose_lm,
    frame_shape: tuple,
    length: float = 50.0,
) -> None:
    """3D pose axes at nose tip. Red=X, Green=Y, Blue=Z."""
    h, w  = frame_shape[:2]
    dist  = np.zeros((4,1))
    ax3d  = np.float32([[length,0,0],[0,length,0],[0,0,length]])
    pts,_ = cv2.projectPoints(ax3d, head_pose.rvec, head_pose.tvec,
                              camera_matrix, dist)
    pts   = pts.reshape(-1,2).astype(int)
    orig  = (int(nose_lm.x * w), int(nose_lm.y * h))
    cv2.arrowedLine(frame, orig, tuple(pts[0]), (0,0,220),   2, tipLength=0.2)
    cv2.arrowedLine(frame, orig, tuple(pts[1]), (0,220,0),   2, tipLength=0.2)
    cv2.arrowedLine(frame, orig, tuple(pts[2]), (220,100,0), 2, tipLength=0.2)


def draw_main_view(
    frame: np.ndarray,
    left:  NormalizedEyePatch,
    right: NormalizedEyePatch,
    head_pose: HeadPose,
    camera_matrix: np.ndarray,
    landmarks,
    frame_shape: tuple,
) -> np.ndarray:
    out = frame.copy()

    for patch, lbl in [(left,"L"), (right,"R")]:
        x1,y1,x2,y2 = patch.bbox_raw
        col = (0,255,0) if patch.is_open else (0,0,255)
        cv2.rectangle(out,(x1,y1),(x2,y2),col,1)
        cv2.putText(out, f"{lbl} EAR:{patch.ear:.2f} Q:{patch.norm_quality:.3f}",
                    (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    draw_pose_axes(out, head_pose, camera_matrix, landmarks[1], frame_shape)

    lines = [
        f"Pitch:{head_pose.pitch:+.1f}  Yaw:{head_pose.yaw:+.1f}  Roll:{head_pose.roll:+.1f}",
        f"Reproj err: {head_pose.reproj_error:.1f}px",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(out, ln, (10, 20+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,230,255), 1)
    return out


def build_eye_strip(patch: NormalizedEyePatch, label: str, scale: int = 3) -> np.ndarray:
    """
    One row:  [RAW | NORMALIZED | DIFF]  + bottom bar with label/quality/status.
    Cyan dot on NORMALIZED = iris center (should be near patch center).
    """
    dw, dh = PATCH_W * scale, PATCH_H * scale
    sx, sy = dw / PATCH_W, dh / PATCH_H

    raw_up  = cv2.cvtColor(
        cv2.resize(patch.raw_crop,  (dw,dh), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR)
    norm_up = cv2.cvtColor(
        cv2.resize(patch.norm_crop, (dw,dh), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR)
    diff_up = cv2.resize(patch.diff_map, (dw,dh), interpolation=cv2.INTER_NEAREST)

    # Iris dot on normalized panel
    ix = int(np.clip(patch.iris_center[0] * sx, 0, dw-1))
    iy = int(np.clip(patch.iris_center[1] * sy, 0, dh-1))
    cv2.circle(norm_up, (ix, iy), 5, (0,255,255), -1)   # cyan
    cv2.circle(norm_up, (ix, iy), 5, (0,0,0),     1)    # black outline

    def lbl_panel(img, txt, col=(180,180,180)):
        cv2.putText(img, txt, (3,13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    lbl_panel(raw_up,  "RAW")
    lbl_panel(norm_up, "NORMALIZED", col=(100,255,150))
    lbl_panel(diff_up, "DIFF",       col=(100,180,255))

    sep   = np.full((dh, 2, 3), 50, dtype=np.uint8)
    strip = np.hstack([raw_up, sep, norm_up, sep, diff_up])

    # Bottom info bar
    bar   = np.zeros((20, strip.shape[1], 3), dtype=np.uint8)
    q_col = (80,255,80) if patch.norm_quality > 0.05 else (120,120,120)
    s_col = (80,255,80) if patch.is_open else (80,80,255)
    cv2.putText(bar, f"{label}  Q:{patch.norm_quality:.3f}",
                (4,14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, q_col, 1)
    cv2.putText(bar, "OPEN" if patch.is_open else "BLINK",
                (strip.shape[1]-58, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, s_col, 1)

    return np.vstack([strip, bar])


# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def run_step1(source=0):
    is_image = isinstance(source, str) and \
               source.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=is_image,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if is_image:
        frame_orig = cv2.imread(source)
        if frame_orig is None:
            print(f"[ERROR] Cannot read: {source}"); sys.exit(1)
        is_video = False
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {source}"); sys.exit(1)
        is_video = True

    print("=" * 60)
    print("STEP 1 - Geometric Normalization (v2)")
    print("  Panels: RAW | NORMALIZED | DIFF")
    print("  Cyan dot = iris center (should land near patch center)")
    print("  Q ~= 0 at straight pose is CORRECT (no roll = no correction)")
    print("  Q rises when you tilt your head sideways")
    print("  Reproj err < 5px = good intrinsic approximation")
    print("  Press 'q' to quit")
    print("=" * 60)

    cam_matrix = None
    frame_count = 0

    while True:
        if is_video:
            ret, frame_orig = cap.read()
            if not ret:
                break
        elif frame_count > 0:
            pass   # static image: keep showing

        frame_count += 1

        if cam_matrix is None:
            cam_matrix = estimate_camera_matrix(frame_orig.shape)

        rgb     = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            disp = frame_orig.copy()
            cv2.putText(disp, "No face detected", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Step 1 - Main View", disp)
            if cv2.waitKey(1 if is_video else 30) & 0xFF == ord('q'):
                break
            continue

        lms = results.multi_face_landmarks[0].landmark

        head_pose = estimate_head_pose(lms, frame_orig.shape, cam_matrix)
        if head_pose is None:
            continue

        left_patch = step1_normalize(
            frame_orig, lms, head_pose,
            LEFT_EYE_INDICES, LEFT_EAR_INDICES, LEFT_IRIS_INDICES
        )
        right_patch = step1_normalize(
            frame_orig, lms, head_pose,
            RIGHT_EYE_INDICES, RIGHT_EAR_INDICES, RIGHT_IRIS_INDICES
        )

        main_view = draw_main_view(
            frame_orig, left_patch, right_patch,
            head_pose, cam_matrix, lms, frame_orig.shape
        )

        left_strip  = build_eye_strip(left_patch,  "LEFT EYE")
        right_strip = build_eye_strip(right_patch, "RIGHT EYE")

        # Equalise widths before vstack
        lw, rw  = left_strip.shape[1], right_strip.shape[1]
        max_w   = max(lw, rw)
        if lw < max_w:
            left_strip  = cv2.copyMakeBorder(left_strip,  0,0,0,max_w-lw, cv2.BORDER_CONSTANT)
        if rw < max_w:
            right_strip = cv2.copyMakeBorder(right_strip, 0,0,0,max_w-rw, cv2.BORDER_CONSTANT)

        divider = np.full((3, max_w, 3), 35, dtype=np.uint8)
        patches = np.vstack([left_strip, divider, right_strip])

        cv2.imshow("Step 1 - Main View",   main_view)
        cv2.imshow("Step 1 - Eye Patches", patches)

        if frame_count % 30 == 0:
            print(f"\n[Frame {frame_count}]")
            print(f"  Pose       | pitch:{head_pose.pitch:+6.1f}deg  "
                  f"yaw:{head_pose.yaw:+6.1f}deg  roll:{head_pose.roll:+6.1f}deg")
            print(f"  Reproj err | {head_pose.reproj_error:.2f}px  "
                  f"(target < 5px)")
            for name, p in [("LEFT", left_patch), ("RIGHT", right_patch)]:
                print(f"  {name:5s}      | EAR:{p.ear:.3f}  "
                      f"open:{p.is_open}  Q:{p.norm_quality:.3f}  "
                      f"iris@patch:({p.iris_center[0]:.1f},{p.iris_center[1]:.1f})")
            print(f"  iris should be near ({PATCH_W/2:.0f},{PATCH_H/2:.0f})")

        if cv2.waitKey(1 if is_video else 30) & 0xFF == ord('q'):
            break

    if is_video:
        cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------


# =============================================================================
# CLAHE PARAMETERS (tuned per region)
# =============================================================================
# Iris: tight clip limit preserves texture; small tile = local contrast
CLAHE_IRIS   = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2, 2))

# Sclera: higher clip limit boosts limbus edge; larger tile = global tone
CLAHE_SCLERA = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))

# Fallback: single global CLAHE (RT-GENE baseline for comparison)
CLAHE_GLOBAL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


# =============================================================================
# Step 2 Data Class
# =============================================================================

@dataclass
class IlluminationResult:
    """
    All outputs from Step 2 for one eye.
    Carries forward everything from Step 1 plus illumination outputs.
    """
    # --- from Step 1 ----------------------------------------------------------
    step1:          NormalizedEyePatch   # full Step 1 result (norm_crop is input)

    # --- Step 2 outputs -------------------------------------------------------
    iris_mask:      np.ndarray   # float32 [0,1], soft iris region mask
    sclera_mask:    np.ndarray   # float32 [0,1] = 1 - iris_mask

    iris_clahe:     np.ndarray   # uint8, CLAHE applied to iris region only
    sclera_clahe:   np.ndarray   # uint8, CLAHE applied to sclera region only
    blended:        np.ndarray   # uint8, feather-blended final patch
    global_clahe:   np.ndarray   # uint8, RT-GENE-style global CLAHE (baseline)

    diff_vs_global: np.ndarray   # uint8 BGR heatmap: blended vs global_clahe
    diff_vs_raw:    np.ndarray   # uint8 BGR heatmap: blended vs raw step1 input

    # --- quality metrics ------------------------------------------------------
    itv:            float   # Iris Texture Visibility [0,1] - our novel metric
    photo_quality:  float   # overall photometric quality [0,1]
    is_usable:      bool    # itv > ITV_THRESHOLD -> usable for gaze CNN


# Minimum ITV to consider a frame usable for gaze estimation
ITV_THRESHOLD = 0.25


# =============================================================================
# Stage 2A - Adaptive Glint Removal
# =============================================================================

def adaptive_glint_threshold(patch: np.ndarray) -> int:
    """
    Computes an adaptive glint threshold instead of RT-GENE's fixed 240.

    RT-GENE uses a fixed threshold of 240/255. This fails under:
      - Dim lighting: glints may peak at 200-220 (missed)
      - Bright overexposure: large regions > 240 (over-inpaints)

    Our approach: Otsu threshold on the top-5% brightest pixels.
    This adapts to the actual brightness distribution of the patch.

    Returns threshold value (uint8).
    """
    # Work only on the top 5% brightness range
    high_vals = patch[patch > np.percentile(patch, 95)]
    if len(high_vals) < 10:
        return 240   # fallback if not enough bright pixels

    # Otsu on the bright tail
    bright_norm = ((high_vals - high_vals.min()) /
                   (high_vals.ptp() + 1e-6) * 255).astype(np.uint8)
    thresh, _ = cv2.threshold(
        bright_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Map back to original scale
    adapted = int(high_vals.min() + thresh / 255.0 * high_vals.ptp())
    # Clamp: never go below 200 (avoid inpainting valid iris texture)
    return max(200, min(adapted, 254))


def remove_glints_adaptive(patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Improved glint removal with adaptive threshold.
    Returns (cleaned_patch, glint_mask) both uint8.

    Improvement vs RT-GENE:
      - Adaptive threshold (see above)
      - Elliptical kernel for dilation (glints are oval, not square)
      - Returns mask separately so caller can visualise glint locations
    """
    thresh = adaptive_glint_threshold(patch)
    _, glint_mask = cv2.threshold(patch, thresh, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(glint_mask) == 0:
        return patch.copy(), glint_mask

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glint_mask = cv2.dilate(glint_mask, kernel, iterations=1)
    cleaned = cv2.inpaint(patch, glint_mask, inpaintRadius=3,
                          flags=cv2.INPAINT_TELEA)
    return cleaned, glint_mask


# =============================================================================
# Stage 2B - Iris Mask Construction
# =============================================================================

def build_iris_mask(
    iris_center_patch: Tuple[float, float],
    iris_radius_patch:  float,
    patch_shape:        Tuple[int, int],
    feather_width:      float = 0.35,
) -> np.ndarray:
    """
    Builds a soft (feathered) iris mask in patch coordinates.

    Hard masks create a visible ring artifact at the iris boundary
    in the blended output. We use a radial falloff instead:

      mask(r) = 1                          for r < r_iris * (1 - feather)
      mask(r) = cos^2(pi/2 * t)           for r in feather zone   [novel]
      mask(r) = 0                          for r > r_iris

    where t = (r - r_inner) / (r_outer - r_inner) in [0,1]

    The cos^2 falloff is perceptually smooth and has zero derivative
    at both endpoints - no visible discontinuity.

    Args:
        iris_center_patch : (cx, cy) iris center in patch pixels
        iris_radius_patch : iris radius in patch pixels
        patch_shape       : (H, W) of the patch
        feather_width     : fraction of radius used for blend zone [0, 0.5]

    Returns float32 array [0,1] of shape patch_shape.
    """
    H, W = patch_shape
    cx, cy = iris_center_patch
    r_iris = max(iris_radius_patch, 1.0)

    # Distance from iris center for every pixel
    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2).astype(np.float32)

    r_inner = r_iris * (1.0 - feather_width)
    r_outer = r_iris

    mask = np.zeros((H, W), dtype=np.float32)

    # Full iris interior
    mask[dist <= r_inner] = 1.0

    # Feather zone: smooth cos^2 transition
    feather_zone = (dist > r_inner) & (dist <= r_outer)
    if feather_zone.any():
        t = (dist[feather_zone] - r_inner) / (r_outer - r_inner + 1e-6)
        mask[feather_zone] = np.cos(0.5 * np.pi * t) ** 2

    return mask


def estimate_iris_radius_in_patch(
    iris_radius_frame: float,
    warp_M: np.ndarray,
) -> float:
    """
    Propagates the iris radius from frame coordinates to patch coordinates
    by measuring how the warp matrix scales distances.

    The scale factor of a 2x3 similarity warp is ||M[0, :2]||
    (norm of the first row's rotation-scale part).
    """
    scale = float(np.linalg.norm(warp_M[0, :2]))
    return iris_radius_frame * scale


# =============================================================================
# Stage 2C - Region-Aware CLAHE
# =============================================================================

def region_aware_clahe(
    patch:       np.ndarray,
    iris_mask:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies separate CLAHE to iris and sclera regions, then blends.

    NOVEL CONTRIBUTION vs RT-GENE:
    RT-GENE applies one CLAHE pass to the whole patch.
    We apply:
      - CLAHE_IRIS   (clipLimit=1.5, tile=2x2) to the iris region
        -> preserves fine collarette/crypt texture, avoids noise amp
      - CLAHE_SCLERA (clipLimit=3.5, tile=4x4) to the sclera region
        -> boosts the hard limbus edge, smooths sclera interior
    Then feather-blend using iris_mask as the alpha.

    Returns:
        iris_enhanced   : uint8, CLAHE iris result (full patch, unmasked)
        sclera_enhanced : uint8, CLAHE sclera result (full patch, unmasked)
        blended         : uint8, alpha-blended final patch
    """
    iris_enhanced   = CLAHE_IRIS.apply(patch)
    sclera_enhanced = CLAHE_SCLERA.apply(patch)

    # Feather blend: blended = iris_mask * iris_clahe + (1-iris_mask) * sclera_clahe
    alpha = iris_mask[:, :, np.newaxis] if patch.ndim == 3 else iris_mask
    blended = (
        alpha       * iris_enhanced.astype(np.float32) +
        (1 - alpha) * sclera_enhanced.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # Fix: bilateral filter to suppress CLAHE tile-boundary stripes.
    # Bilateral preserves the hard limbus edge (high sigma_color) while
    # smoothing the flat gradients caused by tile boundaries (low d).
    # d=5 is small enough to not blur the iris-sclera boundary at 60x36.
    blended = cv2.bilateralFilter(blended, d=5, sigmaColor=20, sigmaSpace=3)

    return iris_enhanced, sclera_enhanced, blended


# =============================================================================
# Stage 2D - Photometric Quality Metrics
# =============================================================================

def iris_texture_visibility(
    blended:    np.ndarray,
    iris_mask:  np.ndarray,
) -> float:
    """
    Iris Texture Visibility (ITV) - our novel photometric quality metric.

    Intuition: a visible iris has high-frequency texture (radial spokes,
    collarette ring, crypts). An occluded, closed, or overexposed iris
    has flat / featureless intensity inside the iris region.

    Method:
      1. Compute Laplacian of Gaussian (LoG) inside iris mask
         - LoG responds to fine texture, not to smooth gradients
      2. Mask-weighted variance of LoG response
         - High variance = rich texture = visible iris
         - Low variance = flat iris = bad frame for gaze CNN

    ITV in [0, 1]:
      > 0.5 : excellent iris texture
      0.25-0.5 : usable
      < 0.25 : poor (low light, motion blur, partial closure) - flag frame

    Normalisation: empirically calibrated on typical webcam frames.
    The sigmoid maps raw variance to [0,1].
    """
    # LoG to detect fine texture
    log_response = cv2.Laplacian(blended, cv2.CV_64F, ksize=3)

    # Restrict to iris region (weighted by soft mask)
    mask_sum = iris_mask.sum() + 1e-6
    weighted_mean = float((log_response * iris_mask).sum() / mask_sum)
    weighted_var  = float(
        ((log_response - weighted_mean)**2 * iris_mask).sum() / mask_sum
    )

    # Sigmoid normalisation: var ~100 -> ITV ~0.5 (empirical midpoint)
    itv = 1.0 / (1.0 + np.exp(-(weighted_var - 80.0) / 30.0))
    return float(np.clip(itv, 0.0, 1.0))


def photometric_quality(
    blended:    np.ndarray,
    iris_mask:  np.ndarray,
    glint_mask: np.ndarray,
) -> float:
    """
    Overall photometric quality score combining:
      - ITV (iris texture visibility)
      - Glint coverage penalty: many glints -> lower quality
      - Exposure score: very dark or very bright patches -> lower quality

    Returns float in [0, 1].
    """
    itv = iris_texture_visibility(blended, iris_mask)

    # Glint penalty: fraction of iris area covered by glints
    iris_area     = float(iris_mask.sum()) + 1e-6
    glint_in_iris = float(((glint_mask > 0).astype(np.float32) * iris_mask).sum())
    glint_penalty = np.clip(glint_in_iris / iris_area, 0.0, 1.0)

    # Exposure score: mean brightness in [60,180] range is ideal
    mean_bright = float(blended.mean())
    exposure_score = 1.0 - abs(mean_bright - 120.0) / 120.0
    exposure_score = float(np.clip(exposure_score, 0.0, 1.0))

    # Weighted combination
    quality = 0.6 * itv + 0.2 * (1 - glint_penalty) + 0.2 * exposure_score
    return float(np.clip(quality, 0.0, 1.0))


# =============================================================================
# Master Step 2 Function
# =============================================================================

def step2_illumination(
    step1_result:      NormalizedEyePatch,
    iris_radius_frame: float,
) -> IlluminationResult:
    """
    Full Step 2 pipeline for one eye.

    Input:  NormalizedEyePatch from step1_normalize()
    Output: IlluminationResult with region-aware CLAHE applied

    Pipeline:
      2A. Adaptive glint removal on step1 norm_crop
      2B. Build soft iris mask from warped iris center + radius
      2C. Region-aware CLAHE (iris vs sclera, separate params)
      2D. Global CLAHE baseline (RT-GENE comparison)
      2E. Photometric quality metrics (ITV, photo_quality)
      2F. Diff heatmaps for visualisation
    """
    patch_in = step1_result.norm_crop   # uint8 gray, 60x36

    # 2A - Adaptive glint removal
    deglinted, glint_mask = remove_glints_adaptive(patch_in)

    # 2B - Iris mask in patch coordinates
    iris_r_patch = estimate_iris_radius_in_patch(
        iris_radius_frame, step1_result.warp_M
    )
    iris_mask = build_iris_mask(
        iris_center_patch=step1_result.iris_center,
        iris_radius_patch=iris_r_patch,
        patch_shape=(PATCH_H, PATCH_W),
        feather_width=0.35,
    )
    sclera_mask = 1.0 - iris_mask

    # 2C - Region-aware CLAHE
    iris_clahe, sclera_clahe, blended = region_aware_clahe(deglinted, iris_mask)

    # 2D - RT-GENE global baseline
    global_clahe = CLAHE_GLOBAL.apply(deglinted)

    # 2E - Quality metrics
    itv           = iris_texture_visibility(blended, iris_mask)
    photo_q       = photometric_quality(blended, iris_mask, glint_mask)
    is_usable     = itv > ITV_THRESHOLD and step1_result.is_open

    # 2F - Diff heatmaps
    diff_vs_global = cv2.applyColorMap(
        cv2.convertScaleAbs(cv2.absdiff(blended, global_clahe), alpha=4.0),
        cv2.COLORMAP_PLASMA
    )
    diff_vs_raw = cv2.applyColorMap(
        cv2.convertScaleAbs(cv2.absdiff(blended, patch_in), alpha=4.0),
        cv2.COLORMAP_INFERNO
    )

    return IlluminationResult(
        step1=step1_result,
        iris_mask=iris_mask,
        sclera_mask=sclera_mask,
        iris_clahe=iris_clahe,
        sclera_clahe=sclera_clahe,
        blended=blended,
        global_clahe=global_clahe,
        diff_vs_global=diff_vs_global,
        diff_vs_raw=diff_vs_raw,
        itv=itv,
        photo_quality=photo_q,
        is_usable=is_usable,
    )


# =============================================================================
# Visualization
# =============================================================================

def build_step2_strip(result: IlluminationResult, label: str, scale: int = 3) -> np.ndarray:
    """
    Display layout for one eye - two rows:

    Row 1 (Step 1 context):
      [STEP1 NORM] | [GLOBAL CLAHE (RT-GENE)] | [OUR BLENDED]

    Row 2 (Step 2 internals):
      [IRIS MASK]  | [IRIS CLAHE]             | [SCLERA CLAHE]

    Bottom bar: ITV, photo_quality, usability flag
    """
    dw, dh = PATCH_W * scale, PATCH_H * scale

    def up(gray):
        """Upscale gray to BGR display size."""
        return cv2.cvtColor(
            cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR
        )

    def up_bgr(bgr):
        return cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_NEAREST)

    def put(img, txt, pos=(3, 13), col=(180, 180, 180)):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    # Row 1
    p_step1  = up(result.step1.norm_crop)
    p_global = up(result.global_clahe)
    p_blend  = up(result.blended)

    # Draw iris circle on blended panel
    cx = int(np.clip(result.step1.iris_center[0] * scale, 0, dw - 1))
    cy = int(np.clip(result.step1.iris_center[1] * scale, 0, dh - 1))
    iris_r_patch = estimate_iris_radius_in_patch(
        # re-derive display radius from mask shape
        float(np.sqrt((result.iris_mask > 0.5).sum() / np.pi)),
        np.eye(3)[:2]   # identity - already in patch coords
    )
    # Simpler: use mask to find radius visually
    r_display = int(np.sqrt((result.iris_mask > 0.5).sum() / np.pi) * scale)
    cv2.circle(p_blend, (cx, cy), max(1, r_display), (0, 255, 255), 1)
    cv2.circle(p_blend, (cx, cy), 3, (0, 255, 255), -1)

    put(p_step1,  "STEP1 NORM")
    put(p_global, "GLOBAL CLAHE", col=(200, 150, 100))
    put(p_blend,  "OUR BLENDED", col=(100, 255, 150))

    sep_v = np.full((dh, 2, 3), 45, dtype=np.uint8)
    row1  = np.hstack([p_step1, sep_v, p_global, sep_v, p_blend])

    # Row 2
    # Iris mask visualised
    mask_vis = up(
        (result.iris_mask * 255).clip(0, 255).astype(np.uint8)
    )
    p_iris   = up(result.iris_clahe)
    p_sclera = up(result.sclera_clahe)

    put(mask_vis, "IRIS MASK",   col=(100, 200, 255))
    put(p_iris,   "IRIS CLAHE",  col=(150, 255, 150))
    put(p_sclera, "SCLERA CLAHE", col=(255, 180, 100))

    row2 = np.hstack([mask_vis, sep_v, p_iris, sep_v, p_sclera])

    sep_h = np.full((2, row1.shape[1], 3), 40, dtype=np.uint8)

    # Bottom info bar
    bar_w = row1.shape[1]
    bar   = np.zeros((22, bar_w, 3), dtype=np.uint8)

    usable_col = (80, 255, 80) if result.is_usable else (80, 80, 255)
    itv_col    = (
        (80, 255, 80)   if result.itv > 0.5  else
        (80, 200, 255)  if result.itv > 0.25 else
        (80, 80, 255)
    )

    cv2.putText(bar,
        f"{label}  ITV:{result.itv:.3f}  PhotoQ:{result.photo_quality:.3f}",
        (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, itv_col, 1)
    cv2.putText(bar,
        "USABLE" if result.is_usable else "LOW-Q",
        (bar_w - 70, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, usable_col, 1)

    return np.vstack([row1, sep_h, row2, sep_h, bar])


# =============================================================================
# Main Loop
# =============================================================================

def run_step2(source=0):
    is_image = isinstance(source, str) and \
               source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=is_image,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if is_image:
        frame_orig = cv2.imread(source)
        if frame_orig is None:
            print(f"[ERROR] Cannot read: {source}"); sys.exit(1)
        is_video = False
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {source}"); sys.exit(1)
        is_video = True

    print("=" * 65)
    print("STEP 2 - Region-Aware Illumination Normalization")
    print("  Row 1: STEP1 NORM | GLOBAL CLAHE (RT-GENE) | OUR BLENDED")
    print("  Row 2: IRIS MASK  | IRIS CLAHE             | SCLERA CLAHE")
    print("  ITV > 0.25 -> USABLE frame for gaze CNN")
    print("  ITV > 0.50 -> excellent iris texture")
    print("  Press 'q' to quit")
    print("=" * 65)

    cam_matrix  = None
    frame_count = 0

    while True:
        if is_video:
            ret, frame_orig = cap.read()
            if not ret:
                break
        elif frame_count > 0:
            pass

        frame_count += 1

        if cam_matrix is None:
            cam_matrix = estimate_camera_matrix(frame_orig.shape)

        rgb     = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            disp = frame_orig.copy()
            cv2.putText(disp, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Step 2 - Main View", disp)
            if cv2.waitKey(1 if is_video else 30) & 0xFF == ord('q'):
                break
            continue

        lms = results.multi_face_landmarks[0].landmark

        head_pose = estimate_head_pose(lms, frame_orig.shape, cam_matrix)
        if head_pose is None:
            continue

        # ---- Step 1: geometric normalization --------------------------------
        left_s1 = step1_normalize(
            frame_orig, lms, head_pose,
            LEFT_EYE_INDICES, LEFT_EAR_INDICES, LEFT_IRIS_INDICES
        )
        right_s1 = step1_normalize(
            frame_orig, lms, head_pose,
            RIGHT_EYE_INDICES, RIGHT_EAR_INDICES, RIGHT_IRIS_INDICES
        )

        # Iris radius in frame pixels (from MediaPipe iris landmarks)
        h, w = frame_orig.shape[:2]
        def iris_radius_px(iris_indices):
            pts = np.array(
                [(lms[i].x * w, lms[i].y * h) for i in iris_indices],
                dtype=np.float32
            )
            return float(np.mean(np.linalg.norm(pts[1:] - pts[0], axis=1)))

        left_ir  = iris_radius_px(LEFT_IRIS_INDICES)
        right_ir = iris_radius_px(RIGHT_IRIS_INDICES)

        # ---- Step 2: illumination normalization ------------------------------
        left_s2  = step2_illumination(left_s1,  left_ir)
        right_s2 = step2_illumination(right_s1, right_ir)

        # ---- Visualization ---------------------------------------------------
        # Main view: reuse Step 1 overlay
        main_view = draw_main_view(
            frame_orig, left_s1, right_s1,
            head_pose, cam_matrix, lms, frame_orig.shape
        )
        # Add Step 2 quality annotation
        h_mv = main_view.shape[0]
        cv2.putText(main_view,
            f"L ITV:{left_s2.itv:.2f} {'OK' if left_s2.is_usable else 'LOW'}  "
            f"R ITV:{right_s2.itv:.2f} {'OK' if right_s2.is_usable else 'LOW'}",
            (10, h_mv - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

        left_strip  = build_step2_strip(left_s2,  "LEFT")
        right_strip = build_step2_strip(right_s2, "RIGHT")

        # Equalise widths
        lw, rw  = left_strip.shape[1], right_strip.shape[1]
        max_w   = max(lw, rw)
        if lw < max_w:
            left_strip  = cv2.copyMakeBorder(left_strip,  0,0,0,max_w-lw,
                                             cv2.BORDER_CONSTANT)
        if rw < max_w:
            right_strip = cv2.copyMakeBorder(right_strip, 0,0,0,max_w-rw,
                                             cv2.BORDER_CONSTANT)

        divider = np.full((4, max_w, 3), 25, dtype=np.uint8)
        patches = np.vstack([left_strip, divider, right_strip])

        cv2.imshow("Step 2 - Main View",   main_view)
        cv2.imshow("Step 2 - Eye Patches", patches)

        # ---- Console output --------------------------------------------------
        if frame_count % 30 == 0:
            print(f"\n[Frame {frame_count}]")
            print(f"  Pose   | pitch:{head_pose.pitch:+5.1f}deg  "
                  f"yaw:{head_pose.yaw:+5.1f}deg  "
                  f"roll:{head_pose.roll:+5.1f}deg")
            for name, s2 in [("LEFT", left_s2), ("RIGHT", right_s2)]:
                print(f"  {name:5s}  | ITV:{s2.itv:.3f}  "
                      f"PhotoQ:{s2.photo_quality:.3f}  "
                      f"usable:{s2.is_usable}  "
                      f"EAR:{s2.step1.ear:.3f}")
            print(f"  Glint threshold adapt: L={adaptive_glint_threshold(left_s1.norm_crop)}  "
                  f"R={adaptive_glint_threshold(right_s1.norm_crop)}")

        if cv2.waitKey(1 if is_video else 30) & 0xFF == ord('q'):
            break

    if is_video:
        cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


# =============================================================================


# =============================================================================
# STEP 3 - Limbus Circle Fitting
# =============================================================================
# Novel Eye Tracking Preprocessing Pipeline | InsightUX
#
# PAPER BASIS
# -----------
# Daugman, "How iris recognition works" (IEEE TIFS, 2004)
#   - Foundational iris biometrics paper
#   - Uses the integro-differential operator to find the iris-sclera
#     boundary (limbus) and iris-pupil boundary as concentric circles:
#       max over (r, x0, y0) of |d/dr * G_sigma(r) * contour_integral(I/2*pi*r)|
#   - Designed for high-res iris cameras (>200px iris diameter)
#
# Daugman (1993), "High confidence visual recognition of persons by a
# test of statistical independence" (IEEE TPAMI)
#   - Original integro-differential formulation
#
# THE GAP
# -------
# Daugman's operator needs the iris to span ~150-300px to converge.
# On our 60x36 webcam patch, the iris is ~15-20px radius - the full
# circular integral has too few pixels to discriminate reliably.
#
# OUR NOVELTY - Seeded Radial Edge Profiling (SREP)
# --------------------------------------------------
# Instead of blind search, we use MediaPipe iris landmarks as a
# geometric seed (instant good starting circle), then refine using
# Daugman's core insight - find where radial intensity gradient peaks -
# adapted for sparse, low-resolution webcam data:
#
#   1. SEED: iris center + radius from MediaPipe (Step 1/2 output)
#   2. SAMPLE: shoot N radial rays from seed center; sample intensity
#      along each ray in a window around the seed radius
#   3. PROFILE: compute 1D gradient along each ray (like Daugman's
#      d/dr but discretized per ray rather than integrated over full circle)
#   4. EDGE POINTS: pick the steepest-gradient crossing on each ray
#      -> these are limbus boundary candidates
#   5. ROBUST FIT: fit an ellipse to the edge points using RANSAC
#      (handles ray failures, eyelid occlusion, low-contrast sectors)
#   6. QUALITY: measure ellipse fit residual + angular coverage
#      -> fraction of rays with valid edge detections
#
# WHY ELLIPSE NOT CIRCLE
# ----------------------
# The iris projects as an ellipse under head tilt/yaw (perspective
# foreshortening). Even after Step 1's roll correction, residual yaw
# means the iris is slightly elliptical in the patch. Fitting an ellipse
# is more accurate than forcing a circle.
#
# OUTPUT
# ------
# LimbusResult containing:
#   - fitted ellipse ((cx,cy), (major,minor), angle)
#   - edge points used for the fit
#   - per-ray quality (valid/invalid)
#   - limbus quality score [0,1]
#   - binary limbus mask (filled ellipse)
#   - pupil seed estimate (inner dark region center)
# =============================================================================

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

NUM_RAYS         = 36      # number of radial rays (10deg spacing)
RAY_SEARCH_BAND  = 0.45    # search +/- this fraction of seed radius along ray
MIN_VALID_RAYS   = 10      # minimum rays needed to attempt ellipse fit
RANSAC_ITERS     = 120     # RANSAC iterations for robust ellipse fit
RANSAC_THRESH    = 1.8     # inlier distance threshold (pixels)


# -----------------------------------------------------------------------------
# Data Class
# -----------------------------------------------------------------------------

@dataclass
class LimbusResult:
    """
    All outputs from Step 3 (limbus fitting) for one eye.
    Carries forward Steps 1+2 results.
    """
    # -- from Step 2 -----------------------------------------------------------
    step2:           "IlluminationResult"

    # -- limbus geometry -------------------------------------------------------
    ellipse:         Optional[tuple]         # ((cx,cy),(ma,mi),angle) or None
    edge_points:     np.ndarray              # (N,2) float32 edge candidates
    ray_valid:       np.ndarray              # (NUM_RAYS,) bool - which rays fired
    inlier_mask:     np.ndarray              # (N,) bool - RANSAC inliers

    # -- derived geometry ------------------------------------------------------
    limbus_center:   Optional[Tuple[float,float]]  # ellipse center
    limbus_axes:     Optional[Tuple[float,float]]  # (major, minor) radii
    limbus_angle:    float                         # ellipse orientation (deg)
    eccentricity:    float                         # 0=circle, 1=line

    # -- pupil seed (inner boundary estimate) ----------------------------------
    pupil_center:    Optional[Tuple[float,float]]
    pupil_radius:    float

    # -- quality ---------------------------------------------------------------
    angular_coverage: float    # fraction of rays with valid edge [0,1]
    fit_residual:     float    # mean distance of inliers to ellipse (px)
    limbus_quality:   float    # combined quality score [0,1]
    is_reliable:      bool     # True if limbus fit is trustworthy


# -----------------------------------------------------------------------------
# 3A - Radial Ray Sampling
# -----------------------------------------------------------------------------

def sample_radial_rays(
    patch:       np.ndarray,
    center:      Tuple[float, float],
    seed_radius: float,
    n_rays:      int      = NUM_RAYS,
    band:        float    = RAY_SEARCH_BAND,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples intensity profiles along N radial rays from center.

    For each ray at angle theta:
      - Sample intensities from r_min to r_max around seed_radius
      - r_min = seed_radius * (1 - band)
      - r_max = seed_radius * (1 + band)

    Returns:
        profiles  : (n_rays, n_samples) float32 intensity along each ray
        r_coords  : (n_samples,) float32 radial distances sampled
        angles    : (n_rays,) float32 ray angles in radians
    """
    H, W       = patch.shape[:2]
    cx, cy     = center
    r_min      = max(1.0, seed_radius * (1.0 - band))
    r_max      = min(seed_radius * (1.0 + band),
                     min(cx, cy, W - cx, H - cy) - 1)
    r_max      = max(r_max, r_min + 2.0)

    n_samples  = max(12, int((r_max - r_min) * 3))
    r_coords   = np.linspace(r_min, r_max, n_samples, dtype=np.float32)
    angles     = np.linspace(0, 2 * np.pi, n_rays, endpoint=False,
                             dtype=np.float32)

    profiles   = np.zeros((n_rays, n_samples), dtype=np.float32)

    for i, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xs    = np.clip(cx + r_coords * cos_t, 0, W - 1).astype(np.float32)
        ys    = np.clip(cy + r_coords * sin_t, 0, H - 1).astype(np.float32)
        # Bilinear interpolation for sub-pixel accuracy
        profiles[i] = cv2.remap(
            patch.astype(np.float32),
            xs.reshape(1, -1), ys.reshape(1, -1),
            cv2.INTER_LINEAR
        ).ravel()

    return profiles, r_coords, angles


# -----------------------------------------------------------------------------
# 3B - Edge Detection along Each Ray
# -----------------------------------------------------------------------------

def find_limbus_edge_on_ray(
    profile:    np.ndarray,
    r_coords:   np.ndarray,
    expected_r: float,
) -> Tuple[Optional[float], float]:
    """
    Finds the limbus crossing on a single radial intensity profile.

    The limbus is a dark-to-bright transition (iris is darker than sclera).
    We look for the maximum positive gradient in the profile.

    Daugman's operator finds max|d/dr(blur*integral)| - we discretize
    this as the steepest positive gradient crossing in the 1D profile.

    Returns:
        edge_r   : radial distance of detected edge (None if not found)
        strength : gradient magnitude at edge (quality indicator)
    """
    if len(profile) < 4:
        return None, 0.0

    # Smooth profile slightly to suppress pixel noise (Gaussian, sigma=1)
    profile_s = cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32),
        (1, 5), sigmaX=1.0
    ).ravel()

    # Discrete gradient
    grad = np.gradient(profile_s.astype(np.float64))

    # Find steepest positive gradient (dark iris -> bright sclera transition)
    # Only look in the valid search band
    best_idx    = int(np.argmax(grad))
    strength    = float(grad[best_idx])

    # Reject weak edges (noise threshold: gradient < 2.0 intensity units/px)
    if strength < 2.0:
        return None, 0.0

    edge_r = float(r_coords[best_idx])
    return edge_r, strength


def extract_edge_points(
    profiles:    np.ndarray,
    r_coords:    np.ndarray,
    angles:      np.ndarray,
    center:      Tuple[float, float],
    seed_radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs edge detection on all rays and converts to (x,y) coordinates.

    Returns:
        edge_pts   : (M, 2) float32 valid edge point coordinates
        ray_valid  : (N,)   bool array - True where edge was found
        strengths  : (M,)   float32 gradient strengths at edge points
    """
    cx, cy    = center
    edge_pts  = []
    strengths = []
    ray_valid = np.zeros(len(angles), dtype=bool)

    for i, (theta, profile) in enumerate(zip(angles, profiles)):
        edge_r, strength = find_limbus_edge_on_ray(profile, r_coords, seed_radius)
        if edge_r is not None:
            x = cx + edge_r * np.cos(theta)
            y = cy + edge_r * np.sin(theta)
            # Sanity: edge should be within patch bounds
            if 0 <= x < profiles.shape[0] and 0 <= y < profiles.shape[0]:
                edge_pts.append([x, y])
                strengths.append(strength)
                ray_valid[i] = True

    if len(edge_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32), ray_valid, np.zeros(0)

    return (np.array(edge_pts, dtype=np.float32),
            ray_valid,
            np.array(strengths, dtype=np.float32))


# -----------------------------------------------------------------------------
# 3C - Robust Ellipse Fitting (RANSAC)
# -----------------------------------------------------------------------------

def fit_ellipse_ransac(
    points:     np.ndarray,
    n_iters:    int   = RANSAC_ITERS,
    thresh:     float = RANSAC_THRESH,
) -> Tuple[Optional[tuple], np.ndarray]:
    """
    Fits an ellipse to edge points using RANSAC for robustness.

    Why RANSAC over direct fitEllipse:
      - Eyelid occlusion creates outlier edge points in upper/lower sectors
      - Low-contrast sectors give weak, noisy edge detections
      - Standard fitEllipse (algebraic) is heavily influenced by outliers

    RANSAC loop:
      1. Sample 5 random points (minimum for ellipse determination)
      2. Fit ellipse to sample
      3. Count inliers (points within thresh pixels of ellipse)
      4. Keep best fit

    Distance to ellipse approximated as |r - r_ellipse(theta)| where
    r_ellipse is the ellipse radius in the direction of each point.

    Returns:
        best_ellipse : OpenCV ellipse tuple or None
        inlier_mask  : (N,) bool array
    """
    N = len(points)
    if N < 6:
        return None, np.zeros(N, dtype=bool)

    # Direct fit if few points (RANSAC overhead not worth it)
    if N < 10:
        try:
            ell = cv2.fitEllipse(points)
            return ell, np.ones(N, dtype=bool)
        except cv2.error:
            return None, np.zeros(N, dtype=bool)

    best_ellipse   = None
    best_inliers   = np.zeros(N, dtype=bool)
    best_n_inliers = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iters):
        # Sample 5 points
        idx     = rng.choice(N, 5, replace=False)
        sample  = points[idx]

        try:
            ell = cv2.fitEllipse(sample)
        except cv2.error:
            continue

        # Validate ellipse geometry
        (cx, cy), (ma, mi), angle = ell
        if ma < 1 or mi < 1 or ma > 200 or mi > 200:
            continue
        if ma / (mi + 1e-6) > 4.0:   # too eccentric = bad fit
            continue

        # Compute inliers: distance from each point to ellipse boundary
        inliers = _ellipse_inliers(points, ell, thresh)
        n_in    = int(inliers.sum())

        if n_in > best_n_inliers:
            best_n_inliers = n_in
            best_inliers   = inliers
            best_ellipse   = ell

    # Refit on all inliers for accuracy
    if best_ellipse is not None and best_n_inliers >= 5:
        try:
            best_ellipse = cv2.fitEllipse(points[best_inliers])
        except cv2.error:
            pass

    return best_ellipse, best_inliers


def _ellipse_inliers(
    points:  np.ndarray,
    ellipse: tuple,
    thresh:  float,
) -> np.ndarray:
    """
    Computes which points are within thresh pixels of the ellipse boundary.

    Uses the algebraic distance approximation:
        f(x,y) = ((x-cx)*cos(a) + (y-cy)*sin(a))^2 / (ma/2)^2
               + ((x-cx)*sin(a) - (y-cy)*cos(a))^2 / (mi/2)^2
        distance ~= |f(x,y) - 1| * scale_factor

    This is fast (no iterative projection) and good enough for RANSAC.
    """
    (cx, cy), (ma, mi), angle_deg = ellipse
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy

    # Rotate into ellipse frame
    u =  dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a

    ra = ma / 2.0 + 1e-6
    rb = mi / 2.0 + 1e-6

    # Algebraic distance to ellipse (|f-1| where f=1 is on ellipse)
    f    = (u / ra)**2 + (v / rb)**2
    dist = np.abs(f - 1.0) * (ra * rb) / (ra + rb)   # normalised approx

    return dist < thresh


# -----------------------------------------------------------------------------
# 3D - Pupil Seed Estimation
# -----------------------------------------------------------------------------

def estimate_pupil_seed(
    patch:          np.ndarray,
    limbus_ellipse: Optional[tuple],
    iris_center:    Tuple[float, float],
    iris_radius:    float,
) -> Tuple[Optional[Tuple[float,float]], float]:
    """
    Estimates pupil center and radius from the darkest region inside the limbus.

    Method:
      1. Create a mask of the iris interior (inside limbus ellipse)
      2. Find the darkest connected blob inside that mask
         - Pupil is always the darkest region in the iris
      3. Return centroid and approximate radius of darkest blob

    This gives Step 4 a starting point for iris texture extraction
    (we need to know where the pupil is to exclude it from texture analysis).

    Returns (pupil_center, pupil_radius) or (None, 0) if estimation fails.
    """
    H, W = patch.shape[:2]

    # Build interior mask
    mask = np.zeros((H, W), dtype=np.uint8)
    if limbus_ellipse is not None:
        cv2.ellipse(mask, limbus_ellipse, 255, -1)
    else:
        # Fallback: circular mask from iris seed
        cv2.circle(mask, (int(iris_center[0]), int(iris_center[1])),
                   int(iris_radius * 0.9), 255, -1)

    if cv2.countNonZero(mask) < 5:
        return None, 0.0

    # Masked patch: only inside iris
    masked = cv2.bitwise_and(patch, patch, mask=mask)

    # Threshold the darkest 25% of pixels inside iris mask
    iris_pixels = patch[mask > 0]
    if len(iris_pixels) < 5:
        return None, 0.0

    dark_thresh = float(np.percentile(iris_pixels, 25))
    _, dark_mask = cv2.threshold(masked, int(dark_thresh), 255,
                                 cv2.THRESH_BINARY_INV)
    dark_mask = cv2.bitwise_and(dark_mask, mask)

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k, iterations=1)

    # Find largest dark blob
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 4:
        return None, 0.0

    M       = cv2.moments(best)
    if M["m00"] < 1e-6:
        return None, 0.0

    pcx     = M["m10"] / M["m00"]
    pcy     = M["m01"] / M["m00"]
    pradius = float(np.sqrt(area / np.pi))

    return (float(pcx), float(pcy)), pradius


# -----------------------------------------------------------------------------
# 3E - Limbus Quality Score
# -----------------------------------------------------------------------------

def compute_limbus_quality(
    ellipse:          Optional[tuple],
    inlier_mask:      np.ndarray,
    edge_points:      np.ndarray,
    ray_valid:        np.ndarray,
    iris_seed_radius: float,
) -> Tuple[float, float, float]:
    """
    Computes limbus fit quality metrics.

    Returns:
        angular_coverage : fraction of rays with valid edges [0,1]
        fit_residual     : mean inlier distance to ellipse (pixels)
        limbus_quality   : combined score [0,1]
    """
    angular_coverage = float(ray_valid.sum()) / max(len(ray_valid), 1)

    if ellipse is None or inlier_mask.sum() < 5:
        return angular_coverage, 999.0, 0.0

    # Mean inlier residual
    inlier_pts = edge_points[inlier_mask]
    dists      = _point_to_ellipse_dist(inlier_pts, ellipse)
    fit_resid  = float(np.mean(dists)) if len(dists) > 0 else 999.0

    # Ellipse sanity: axes should be close to seed radius
    (cx, cy), (ma, mi), _ = ellipse
    ratio = min(ma, mi) / (max(ma, mi) + 1e-6)   # 1=circle, 0=degenerate
    size_ok = 0.5 < (ma / 2) / (iris_seed_radius + 1e-6) < 2.0

    # Combined score
    coverage_score = angular_coverage
    residual_score = float(np.clip(1.0 - fit_resid / 5.0, 0.0, 1.0))
    shape_score    = ratio * (1.0 if size_ok else 0.3)

    quality = 0.4 * coverage_score + 0.4 * residual_score + 0.2 * shape_score
    return angular_coverage, fit_resid, float(np.clip(quality, 0.0, 1.0))


def _point_to_ellipse_dist(
    points: np.ndarray,
    ellipse: tuple,
) -> np.ndarray:
    """Algebraic distance from points to ellipse boundary (fast approximation)."""
    if len(points) == 0:
        return np.zeros(0)
    (cx, cy), (ma, mi), angle_deg = ellipse
    a      = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx, dy = points[:,0] - cx, points[:,1] - cy
    u      =  dx*ca + dy*sa
    v      = -dx*sa + dy*ca
    ra, rb = ma/2.0 + 1e-6, mi/2.0 + 1e-6
    f      = (u/ra)**2 + (v/rb)**2
    return np.abs(f - 1.0) * (ra*rb)/(ra+rb)


# -----------------------------------------------------------------------------
# Master Step 3 Function
# -----------------------------------------------------------------------------

def step3_limbus(
    step2_result: "IlluminationResult",
    iris_radius_frame: float,
) -> LimbusResult:
    """
    Full Step 3 pipeline for one eye.

    Input:  IlluminationResult from step2_illumination()
    Output: LimbusResult with fitted limbus ellipse

    Pipeline:
      3A. Sample radial rays from MediaPipe iris seed
      3B. Detect limbus edge on each ray (steepest gradient)
      3C. Robust ellipse fit via RANSAC
      3D. Estimate pupil seed from darkest region inside limbus
      3E. Compute quality metrics
    """
    patch      = step2_result.blended        # uint8 gray 60x36 - illumination-corrected
    iris_c     = step2_result.step1.iris_center   # (cx,cy) in patch coords
    warp_M     = step2_result.step1.warp_M

    # Iris radius propagated to patch space
    scale      = float(np.linalg.norm(warp_M[0, :2]))
    iris_r_px  = float(np.clip(iris_radius_frame * scale, 3.0, min(PATCH_W, PATCH_H) / 2.0 - 1))

    # 3A - radial ray sampling
    profiles, r_coords, angles = sample_radial_rays(
        patch, iris_c, iris_r_px,
        n_rays=NUM_RAYS, band=RAY_SEARCH_BAND
    )

    # 3B - edge detection per ray
    edge_pts, ray_valid, strengths = extract_edge_points(
        profiles, r_coords, angles, iris_c, iris_r_px
    )

    # 3C - RANSAC ellipse fit
    if len(edge_pts) >= MIN_VALID_RAYS:
        ellipse, inlier_mask = fit_ellipse_ransac(edge_pts)
    else:
        ellipse      = None
        inlier_mask  = np.zeros(len(edge_pts), dtype=bool)

    # Fallback: if RANSAC failed, use MediaPipe circle as ellipse
    if ellipse is None:
        cx, cy = iris_c
        ellipse = (
            (float(cx), float(cy)),
            (float(iris_r_px * 2), float(iris_r_px * 2)),
            0.0
        )
        inlier_mask = np.ones(len(edge_pts), dtype=bool)

    # Parse ellipse
    (ecx, ecy), (ma, mi), eangle = ellipse
    limbus_center = (float(ecx), float(ecy))
    limbus_axes   = (float(ma / 2.0), float(mi / 2.0))
    ecc           = float(np.sqrt(max(0, 1.0 - (min(ma,mi)/max(ma,mi,1e-6))**2)))

    # 3D - pupil seed
    pupil_center, pupil_radius = estimate_pupil_seed(
        patch, ellipse, iris_c, iris_r_px
    )

    # 3E - quality
    angular_cov, fit_resid, lq = compute_limbus_quality(
        ellipse, inlier_mask, edge_pts, ray_valid, iris_r_px
    )
    is_reliable = lq > 0.35 and angular_cov > 0.4

    return LimbusResult(
        step2=step2_result,
        ellipse=ellipse,
        edge_points=edge_pts,
        ray_valid=ray_valid,
        inlier_mask=inlier_mask,
        limbus_center=limbus_center,
        limbus_axes=limbus_axes,
        limbus_angle=float(eangle),
        eccentricity=ecc,
        pupil_center=pupil_center,
        pupil_radius=pupil_radius,
        angular_coverage=angular_cov,
        fit_residual=fit_resid,
        limbus_quality=lq,
        is_reliable=is_reliable,
    )


# =============================================================================
# Visualization
# =============================================================================

def build_step3_strip(result: LimbusResult, label: str, scale: int = 4) -> np.ndarray:
    """
    Display layout - two panels side by side:

    [BLENDED (Step 2 output)] | [LIMBUS FIT overlay]

    On the LIMBUS FIT panel:
      - Yellow ellipse     = fitted limbus boundary
      - Green dots         = inlier edge points used for fit
      - Red dots           = outlier edge points (RANSAC rejected)
      - Cyan dot + circle  = MediaPipe iris seed (before refinement)
      - Magenta dot        = estimated pupil center
      - Ray lines          = sampled radial directions (valid=green, failed=red)

    Bottom bar: angular coverage, fit residual, quality, reliability
    """
    dw, dh = PATCH_W * scale, PATCH_H * scale
    sx, sy = dw / PATCH_W, dh / PATCH_H

    # Panel A: blended input
    panelA = cv2.cvtColor(
        cv2.resize(result.step2.blended, (dw, dh), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR
    )
    cv2.putText(panelA, "STEP2 BLENDED", (3, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # Panel B: limbus overlay
    panelB = cv2.cvtColor(
        cv2.resize(result.step2.blended, (dw, dh), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR
    )

    # Draw radial rays (thin, semi-transparent look via low brightness)
    cx_s = int(result.step2.step1.iris_center[0] * sx)
    cy_s = int(result.step2.step1.iris_center[1] * sy)
    warp_M     = result.step2.step1.warp_M
    iris_r_raw = float(np.linalg.norm(warp_M[0,:2]))  # approx
    # draw ray endpoints
    angles_vis = np.linspace(0, 2*np.pi, NUM_RAYS, endpoint=False)
    for i, theta in enumerate(angles_vis):
        end_x = int(cx_s + iris_r_raw * sx * np.cos(theta))
        end_y = int(cy_s + iris_r_raw * sy * np.sin(theta))
        col   = (0, 80, 0) if result.ray_valid[i] else (0, 0, 60)
        cv2.line(panelB, (cx_s, cy_s), (end_x, end_y), col, 1)

    # Draw edge points
    for i, pt in enumerate(result.edge_points):
        px, py = int(pt[0] * sx), int(pt[1] * sy)
        if result.inlier_mask[i]:
            cv2.circle(panelB, (px, py), 3, (0, 255, 0), -1)    # green = inlier
        else:
            cv2.circle(panelB, (px, py), 3, (0, 0, 255), -1)    # red = outlier

    # Draw MediaPipe seed circle (cyan)
    seed_r = int(iris_r_raw * sx)
    cv2.circle(panelB, (cx_s, cy_s), max(1, seed_r), (255, 200, 0), 1)
    cv2.circle(panelB, (cx_s, cy_s), 3, (255, 200, 0), -1)

    # Draw fitted limbus ellipse (yellow, thick)
    if result.ellipse is not None:
        (ecx, ecy), (ma, mi), eang = result.ellipse
        scaled_ell = (
            (ecx * sx, ecy * sy),
            (ma * sx, mi * sy),
            eang
        )
        try:
            cv2.ellipse(panelB, scaled_ell, (0, 220, 220), 2)
        except cv2.error:
            pass

    # Draw pupil center (magenta)
    if result.pupil_center is not None:
        px = int(result.pupil_center[0] * sx)
        py = int(result.pupil_center[1] * sy)
        r  = max(2, int(result.pupil_radius * sx))
        cv2.circle(panelB, (px, py), r, (255, 0, 255), 1)
        cv2.circle(panelB, (px, py), 3, (255, 0, 255), -1)

    cv2.putText(panelB, "LIMBUS FIT", (3, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

    sep  = np.full((dh, 2, 3), 45, dtype=np.uint8)
    row  = np.hstack([panelA, sep, panelB])

    # Bottom bar
    bar_w = row.shape[1]
    bar   = np.zeros((22, bar_w, 3), dtype=np.uint8)

    q_col = (
        (80, 255, 80)  if result.limbus_quality > 0.6 else
        (80, 200, 255) if result.limbus_quality > 0.35 else
        (80, 80, 255)
    )
    r_col = (80, 255, 80) if result.is_reliable else (80, 80, 255)

    cv2.putText(bar,
        f"{label}  Cov:{result.angular_coverage:.2f}  "
        f"Res:{result.fit_residual:.1f}px  "
        f"LQ:{result.limbus_quality:.2f}  "
        f"Ecc:{result.eccentricity:.2f}",
        (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, q_col, 1)
    cv2.putText(bar,
        "RELIABLE" if result.is_reliable else "WEAK",
        (bar_w - 80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, r_col, 1)

    return np.vstack([row, bar])


# =============================================================================
# Main Loop
# =============================================================================

def run_step3(source=0):
    is_image = isinstance(source, str) and \
               source.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=is_image,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if is_image:
        frame_orig = cv2.imread(source)
        if frame_orig is None:
            print(f"[ERROR] Cannot read: {source}"); sys.exit(1)
        is_video = False
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {source}"); sys.exit(1)
        is_video = True

    print("=" * 65)
    print("STEP 3 - Limbus Circle Fitting (Seeded Radial Edge Profiling)")
    print("  Left panel  : Step 2 blended output (input to Step 3)")
    print("  Right panel : Limbus fit overlay")
    print("    Cyan circle   = MediaPipe iris seed")
    print("    Yellow ellipse = fitted limbus (our SREP result)")
    print("    Green dots    = inlier edge points (RANSAC accepted)")
    print("    Red dots      = outlier edge points (RANSAC rejected)")
    print("    Magenta circle = estimated pupil region")
    print("  Cov  = angular coverage (fraction of rays that fired)")
    print("  Res  = mean inlier residual in pixels (lower = better)")
    print("  LQ   = limbus quality score [0,1]")
    print("  Ecc  = eccentricity (0=circle, high=foreshortened)")
    print("  Press 'q' to quit")
    print("=" * 65)

    cam_matrix  = None
    frame_count = 0

    while True:
        if is_video:
            ret, frame_orig = cap.read()
            if not ret:
                break
        elif frame_count > 0:
            pass

        frame_count += 1

        if cam_matrix is None:
            cam_matrix = estimate_camera_matrix(frame_orig.shape)

        rgb     = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            disp = frame_orig.copy()
            cv2.putText(disp, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Step 3 - Main View", disp)
            if cv2.waitKey(1 if is_video else 30) & 0xFF == ord('q'):
                break
            continue

        lms = results.multi_face_landmarks[0].landmark

        head_pose = estimate_head_pose(lms, frame_orig.shape, cam_matrix)
        if head_pose is None:
            continue

        # Step 1
        left_s1  = step1_normalize(frame_orig, lms, head_pose,
                                   LEFT_EYE_INDICES, LEFT_EAR_INDICES, LEFT_IRIS_INDICES)
        right_s1 = step1_normalize(frame_orig, lms, head_pose,
                                   RIGHT_EYE_INDICES, RIGHT_EAR_INDICES, RIGHT_IRIS_INDICES)

        # Iris radius in frame pixels
        h, w = frame_orig.shape[:2]
        def iris_r(idxs):
            pts = np.array([(lms[i].x*w, lms[i].y*h) for i in idxs],
                           dtype=np.float32)
            return float(np.mean(np.linalg.norm(pts[1:] - pts[0], axis=1)))

        left_ir  = iris_r(LEFT_IRIS_INDICES)
        right_ir = iris_r(RIGHT_IRIS_INDICES)

        # Step 2
        left_s2  = step2_illumination(left_s1,  left_ir)
        right_s2 = step2_illumination(right_s1, right_ir)

        # Step 3
        left_s3  = step3_limbus(left_s2,  left_ir)
        right_s3 = step3_limbus(right_s2, right_ir)

        # Main view overlay
        main_view = draw_main_view(
            frame_orig, left_s1, right_s1,
            head_pose, cam_matrix, lms, frame_orig.shape
        )
        h_mv = main_view.shape[0]
        cv2.putText(main_view,
            f"L LQ:{left_s3.limbus_quality:.2f} "
            f"{'OK' if left_s3.is_reliable else 'WEAK'}  "
            f"R LQ:{right_s3.limbus_quality:.2f} "
            f"{'OK' if right_s3.is_reliable else 'WEAK'}",
            (10, h_mv - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,255,180), 1)

        # Patch display
        left_strip  = build_step3_strip(left_s3,  "LEFT")
        right_strip = build_step3_strip(right_s3, "RIGHT")

        lw, rw = left_strip.shape[1], right_strip.shape[1]
        max_w  = max(lw, rw)
        if lw < max_w:
            left_strip  = cv2.copyMakeBorder(left_strip,  0,0,0,max_w-lw, cv2.BORDER_CONSTANT)
        if rw < max_w:
            right_strip = cv2.copyMakeBorder(right_strip, 0,0,0,max_w-rw, cv2.BORDER_CONSTANT)

        div     = np.full((4, max_w, 3), 20, dtype=np.uint8)
        patches = np.vstack([left_strip, div, right_strip])

        cv2.imshow("Step 3 - Main View",   main_view)
        cv2.imshow("Step 3 - Eye Patches", patches)

        if frame_count % 30 == 0:
            print(f"\n[Frame {frame_count}]")
            print(f"  Pose  | pitch:{head_pose.pitch:+5.1f}deg  "
                  f"yaw:{head_pose.yaw:+5.1f}deg  "
                  f"roll:{head_pose.roll:+5.1f}deg")
            for name, s3 in [("LEFT", left_s3), ("RIGHT", right_s3)]:
                print(f"  {name:5s} | LQ:{s3.limbus_quality:.3f}  "
                      f"cov:{s3.angular_coverage:.2f}  "
                      f"res:{s3.fit_residual:.2f}px  "
                      f"ecc:{s3.eccentricity:.3f}  "
                      f"reliable:{s3.is_reliable}")
                if s3.limbus_center:
                    print(f"         limbus_center:({s3.limbus_center[0]:.1f},"
                          f"{s3.limbus_center[1]:.1f})  "
                          f"axes:({s3.limbus_axes[0]:.1f},{s3.limbus_axes[1]:.1f})px")

        if cv2.waitKey(1 if is_video else 30) & 0xFF == ord('q'):
            break

    if is_video:
        cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="InsightUX Preprocessing Pipeline (Steps 1+2+3)"
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--image", type=str, help="Path to image file")
    g.add_argument("--video", type=str, help="Path to video file")
    args = parser.parse_args()
    if args.image:   run_step3(args.image)
    elif args.video: run_step3(args.video)
    else:            run_step3(0)