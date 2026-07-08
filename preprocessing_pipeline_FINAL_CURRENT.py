"""
COMPLETE PATCHED preprocessing_pipeline.py
ALL FIXES INTEGRATED:
  - Anisotropic foreshortening correction in compute_roi_warp()
  - Updated step1_normalize() to pass head_pose
  - Fixed bounds check in extract_edge_points()

STEP 2 - Region-Aware Illumination Normalization
=================================================
Novel Eye Tracking Preprocessing Pipeline | InsightUX

Run AFTER step1_geometric_normalization.py.
This file imports Step 1 and adds Step 2 on top.
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
#
# FIX (head-pose convention bug): the original model used a Y-up
# convention (chin negative, eyes positive), which conflicts with
# OpenCV's camera convention (Y-down, Z-forward). That mismatch made
# solvePnP report a baked-in ~180deg pitch offset for a straight,
# frontal face (observed as head_pose.pitch landing near -177deg
# instead of a sane small value). Y and Z signs are flipped here so
# the model matches OpenCV's convention: a frontal face now yields
# head_pose.pitch/yaw/roll near 0, as it should.
FACE_3D_MODEL = np.array([
    [  0.0,    0.0,    0.0],   # nose tip         -> landmark 1
    [  0.0,   63.6,   12.5],  # chin             -> landmark 152
    [-43.3,  -32.7,   26.0],  # left eye outer   -> landmark 33
    [ 43.3,  -32.7,   26.0],  # right eye outer  -> landmark 263
    [-28.9,   28.9,   24.1],  # left mouth       -> landmark 61
    [ 28.9,   28.9,   24.1],  # right mouth      -> landmark 291
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
    pitch:        float
    yaw:          float
    roll:         float
    rvec:         np.ndarray
    tvec:         np.ndarray
    reproj_error: float


@dataclass
class NormalizedEyePatch:
    raw_crop:     np.ndarray
    norm_crop:    np.ndarray
    diff_map:     np.ndarray
    bbox_raw:     Tuple[int,int,int,int]
    warp_M:       np.ndarray
    iris_center:  Tuple[float,float]
    ear:          float
    is_open:      bool
    norm_quality: float


# -----------------------------------------------------------------------------
# Camera Intrinsics
# -----------------------------------------------------------------------------

def estimate_camera_matrix(frame_shape: tuple) -> np.ndarray:
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
    """Rotation matrix -> (pitch, yaw, roll) degrees."""
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


# =============================================================================
# 2D Normalization Warp (PATCHED WITH FORESHORTENING CORRECTION)
# =============================================================================

def compute_roi_warp(
    landmarks,
    frame_shape: tuple,
    bbox: Tuple[int,int,int,int],
    anchor_indices,
    roll_deg: float,
    head_pose: 'HeadPose' = None,
) -> Tuple[np.ndarray, Tuple[float,float]]:
    """
    Compute 2D affine warp for eye ROI normalization, with optional
    anisotropic foreshortening correction.
    
    PATCHED: Now includes foreshortening correction via anisotropic pre-scaling.
    
    Args:
        head_pose: optional HeadPose object. If provided, applies anisotropic
                   pre-scaling to approximately undo the eye shape distortion
                   caused by head pitch/yaw. If None, uses the old (roll-only)
                   normalization.
    
    Returns:
        (warp_M, anchor_local): affine warp matrix and anchor point in ROI coords
    
    The foreshortening correction works by:
      - A head pitched down (looking at screen) makes the eye appear taller
        in the image (Y foreshortened) -> pre-scale X by 1/cos(pitch)
      - A head turned left (yaw negative) makes the eye appear narrower
        in the image (X foreshortened) -> pre-scale Y by 1/cos(yaw)
    
    Both corrections are applied as pre-scaling before the existing similarity
    (rotate + translate + uniform scale) warp, so they don't interfere with
    roll correction.
    """
    h, w   = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    roi_w  = max(x2 - x1, 1)
    roi_h  = max(y2 - y1, 1)

    anchor_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in anchor_indices],
        dtype=np.float32
    )
    anchor_global = anchor_pts.mean(axis=0)

    anchor_local = (
        float(anchor_global[0] - x1),
        float(anchor_global[1] - y1),
    )

    # ===== ANISOTROPIC PRE-SCALING FOR FORESHORTENING CORRECTION (PATCHED) =====
    prescale_x = 1.0
    prescale_y = 1.0
    
    if head_pose is not None:
        # Pitch correction: pitched-down face makes eye appear taller (Y compressed)
        # -> pre-scale X to compensate
        pitch_rad = np.radians(np.clip(head_pose.pitch, -35.0, 35.0))
        prescale_x *= 1.0 / (np.cos(pitch_rad) + 1e-6)
        
        # Yaw correction: turned-away face makes eye appear narrower (X compressed)
        # -> pre-scale Y to compensate
        yaw_rad = np.radians(np.clip(head_pose.yaw, -35.0, 35.0))
        prescale_y *= 1.0 / (np.cos(yaw_rad) + 1e-6)
        
        # Clip to prevent extreme scaling (edge case: head >45° off-axis)
        prescale_x = np.clip(prescale_x, 0.8, 1.5)
        prescale_y = np.clip(prescale_y, 0.8, 1.5)
    # =========================================================================

    # Effective ROI dimensions after pre-scaling
    roi_w_eff = roi_w * prescale_x
    roi_h_eff = roi_h * prescale_y
    
    scale = min(PATCH_W / roi_w_eff, PATCH_H / roi_h_eff)

    a = np.radians(-roll_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)

    cx, cy = anchor_local
    
    # Apply pre-scaling to anchor point before rotation/translation
    cx_prescaled = cx * prescale_x
    cy_prescaled = cy * prescale_y

    tx = PATCH_W / 2.0 + scale * (-cos_a * cx_prescaled + sin_a * cy_prescaled)
    ty = PATCH_H / 2.0 + scale * (-sin_a * cx_prescaled - cos_a * cy_prescaled)

    warp_M = np.array([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a,  scale * cos_a, ty],
    ], dtype=np.float64)

    return warp_M, anchor_local


def apply_warp_to_roi(
    gray_roi: np.ndarray,
    warp_M:   np.ndarray,
) -> np.ndarray:
    return cv2.warpAffine(
        gray_roi, warp_M, (PATCH_W, PATCH_H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )


def raw_crop_resized(gray_roi: np.ndarray) -> np.ndarray:
    if gray_roi.size == 0:
        return np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
    return cv2.resize(gray_roi, (PATCH_W, PATCH_H), interpolation=cv2.INTER_LANCZOS4)


# =============================================================================
# Iris Center in Normalized Patch Coords
# =============================================================================

def iris_center_in_patch(
    iris_local: Tuple[float,float],
    warp_M: np.ndarray,
) -> Tuple[float,float]:
    ix, iy = iris_local
    px = warp_M[0,0]*ix + warp_M[0,1]*iy + warp_M[0,2]
    py = warp_M[1,0]*ix + warp_M[1,1]*iy + warp_M[1,2]
    return float(px), float(py)


# =============================================================================
# Normalization Quality Metric
# =============================================================================

def compute_norm_quality(
    raw_crop:  np.ndarray,
    norm_crop: np.ndarray,
) -> float:
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


# =============================================================================
# EAR
# =============================================================================

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


# =============================================================================
# Master Step 1 Function (PATCHED TO PASS head_pose)
# =============================================================================

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
    PATCHED: Now passes full head_pose to compute_roi_warp for foreshortening correction.
    """
    ear     = compute_ear(landmarks, ear_indices, frame.shape)
    is_open = ear >= ear_threshold

    gray_roi, bbox = extract_roi(frame, landmarks, eye_indices)

    # ANCHOR FIX: warp now centers on the eye-corner midpoint, not the
    # iris. ear_indices[0] and ear_indices[3] are the outer and inner
    # canthus (eye corner) landmarks - see compute_ear, where they're
    # used as the EAR denominator (eye width).
    #
    # PATCHED: Also pass full head_pose to compute_roi_warp for foreshortening
    # correction (was only passing roll before)
    canthus_indices = (ear_indices[0], ear_indices[3])
    warp_M, _anchor_local = compute_roi_warp(
        landmarks, frame.shape, bbox, canthus_indices, head_pose.roll,
        head_pose=head_pose  # PATCHED: pass full head pose for foreshortening correction
    )

    norm_crop = apply_warp_to_roi(gray_roi, warp_M)
    raw_crop = raw_crop_resized(gray_roi)

    diff      = cv2.absdiff(raw_crop, norm_crop)
    diff_vis  = cv2.applyColorMap(
        cv2.convertScaleAbs(diff, alpha=4.0), cv2.COLORMAP_INFERNO
    )

    # Iris position in ROI-local coords, projected through the SAME warp.
    # Still needed for Step 2's region-aware CLAHE (iris vs sclera mask).
    # It now lands at a different spot in the patch depending on gaze
    # direction, instead of always sitting dead-center.
    h, w = frame.shape[:2]
    iris_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices],
        dtype=np.float32
    )
    iris_global = iris_pts[0]
    x1, y1, _, _ = bbox
    iris_local = (float(iris_global[0] - x1), float(iris_global[1] - y1))
    iris_patch = iris_center_in_patch(iris_local, warp_M)

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


# =============================================================================
# Visualization helpers (used only by the standalone run_step1/2/3 viewers)
# =============================================================================

def draw_pose_axes(
    frame: np.ndarray,
    head_pose: HeadPose,
    camera_matrix: np.ndarray,
    nose_lm,
    frame_shape: tuple,
    length: float = 50.0,
) -> None:
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


# =============================================================================
# CLAHE PARAMETERS (tuned per region)
# =============================================================================
CLAHE_IRIS   = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2, 2))
CLAHE_SCLERA = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
CLAHE_GLOBAL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


# =============================================================================
# Step 2 Data Class
# =============================================================================

@dataclass
class IlluminationResult:
    step1:          NormalizedEyePatch
    iris_mask:      np.ndarray
    sclera_mask:    np.ndarray
    iris_clahe:     np.ndarray
    sclera_clahe:   np.ndarray
    blended:        np.ndarray
    global_clahe:   np.ndarray
    diff_vs_global: np.ndarray
    diff_vs_raw:    np.ndarray
    itv:            float
    photo_quality:  float
    is_usable:      bool


ITV_THRESHOLD = 0.25


# =============================================================================
# Stage 2A - Adaptive Glint Removal
# =============================================================================

def adaptive_glint_threshold(patch: np.ndarray) -> int:
    high_vals = patch[patch > np.percentile(patch, 95)]
    if len(high_vals) < 10:
        return 240

    bright_norm = ((high_vals - high_vals.min()) /
                   (high_vals.ptp() + 1e-6) * 255).astype(np.uint8)
    thresh, _ = cv2.threshold(
        bright_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adapted = int(high_vals.min() + thresh / 255.0 * high_vals.ptp())
    return max(200, min(adapted, 254))


def remove_glints_adaptive(patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    H, W = patch_shape
    cx, cy = iris_center_patch
    r_iris = max(iris_radius_patch, 1.0)

    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2).astype(np.float32)

    r_inner = r_iris * (1.0 - feather_width)
    r_outer = r_iris

    mask = np.zeros((H, W), dtype=np.float32)
    mask[dist <= r_inner] = 1.0

    feather_zone = (dist > r_inner) & (dist <= r_outer)
    if feather_zone.any():
        t = (dist[feather_zone] - r_inner) / (r_outer - r_inner + 1e-6)
        mask[feather_zone] = np.cos(0.5 * np.pi * t) ** 2

    return mask


def estimate_iris_radius_in_patch(
    iris_radius_frame: float,
    warp_M: np.ndarray,
) -> float:
    scale = float(np.linalg.norm(warp_M[0, :2]))
    return iris_radius_frame * scale


# =============================================================================
# Stage 2C - Region-Aware CLAHE
# =============================================================================

def region_aware_clahe(
    patch:       np.ndarray,
    iris_mask:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iris_enhanced   = CLAHE_IRIS.apply(patch)
    sclera_enhanced = CLAHE_SCLERA.apply(patch)

    alpha = iris_mask[:, :, np.newaxis] if patch.ndim == 3 else iris_mask
    blended = (
        alpha       * iris_enhanced.astype(np.float32) +
        (1 - alpha) * sclera_enhanced.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    blended = cv2.bilateralFilter(blended, d=5, sigmaColor=20, sigmaSpace=3)

    return iris_enhanced, sclera_enhanced, blended


# =============================================================================
# Stage 2D - Photometric Quality Metrics
# =============================================================================

def iris_texture_visibility(
    blended:    np.ndarray,
    iris_mask:  np.ndarray,
) -> float:
    log_response = cv2.Laplacian(blended, cv2.CV_64F, ksize=3)

    mask_sum = iris_mask.sum() + 1e-6
    weighted_mean = float((log_response * iris_mask).sum() / mask_sum)
    weighted_var  = float(
        ((log_response - weighted_mean)**2 * iris_mask).sum() / mask_sum
    )

    itv = 1.0 / (1.0 + np.exp(-(weighted_var - 80.0) / 30.0))
    return float(np.clip(itv, 0.0, 1.0))


def photometric_quality(
    blended:    np.ndarray,
    iris_mask:  np.ndarray,
    glint_mask: np.ndarray,
) -> float:
    itv = iris_texture_visibility(blended, iris_mask)

    iris_area     = float(iris_mask.sum()) + 1e-6
    glint_in_iris = float(((glint_mask > 0).astype(np.float32) * iris_mask).sum())
    glint_penalty = np.clip(glint_in_iris / iris_area, 0.0, 1.0)

    mean_bright = float(blended.mean())
    exposure_score = 1.0 - abs(mean_bright - 120.0) / 120.0
    exposure_score = float(np.clip(exposure_score, 0.0, 1.0))

    quality = 0.6 * itv + 0.2 * (1 - glint_penalty) + 0.2 * exposure_score
    return float(np.clip(quality, 0.0, 1.0))


# =============================================================================
# Master Step 2 Function
# =============================================================================

def step2_illumination(
    step1_result:      NormalizedEyePatch,
    iris_radius_frame: float,
) -> IlluminationResult:
    patch_in = step1_result.norm_crop

    deglinted, glint_mask = remove_glints_adaptive(patch_in)

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

    iris_clahe, sclera_clahe, blended = region_aware_clahe(deglinted, iris_mask)

    global_clahe = CLAHE_GLOBAL.apply(deglinted)

    itv           = iris_texture_visibility(blended, iris_mask)
    photo_q       = photometric_quality(blended, iris_mask, glint_mask)
    is_usable     = itv > ITV_THRESHOLD and step1_result.is_open

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
# STEP 3 - Limbus Circle Fitting (Seeded Radial Edge Profiling)
# =============================================================================

from dataclasses import field
from typing import List

NUM_RAYS         = 36
RAY_SEARCH_BAND  = 0.45
MIN_VALID_RAYS   = 10
RANSAC_ITERS     = 120
RANSAC_THRESH    = 1.8


@dataclass
class LimbusResult:
    step2:           "IlluminationResult"
    ellipse:         Optional[tuple]
    edge_points:     np.ndarray
    ray_valid:       np.ndarray
    inlier_mask:     np.ndarray
    limbus_center:   Optional[Tuple[float,float]]
    limbus_axes:     Optional[Tuple[float,float]]
    limbus_angle:    float
    eccentricity:    float
    pupil_center:    Optional[Tuple[float,float]]
    pupil_radius:    float
    angular_coverage: float
    fit_residual:     float
    limbus_quality:   float
    is_reliable:      bool


def sample_radial_rays(
    patch:       np.ndarray,
    center:      Tuple[float, float],
    seed_radius: float,
    n_rays:      int      = NUM_RAYS,
    band:        float    = RAY_SEARCH_BAND,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        profiles[i] = cv2.remap(
            patch.astype(np.float32),
            xs.reshape(1, -1), ys.reshape(1, -1),
            cv2.INTER_LINEAR
        ).ravel()

    return profiles, r_coords, angles


def find_limbus_edge_on_ray(
    profile:    np.ndarray,
    r_coords:   np.ndarray,
    expected_r: float,
) -> Tuple[Optional[float], float]:
    if len(profile) < 4:
        return None, 0.0

    profile_s = cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32),
        (1, 5), sigmaX=1.0
    ).ravel()

    grad = np.gradient(profile_s.astype(np.float64))

    best_idx    = int(np.argmax(grad))
    strength    = float(grad[best_idx])

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
    PATCHED: Fixed bounds check to use PATCH_W and PATCH_H instead of profiles.shape[0]
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
            # PATCHED: Use PATCH_W and PATCH_H instead of profiles.shape[0]
            if 0 <= x < PATCH_W and 0 <= y < PATCH_H:
                edge_pts.append([x, y])
                strengths.append(strength)
                ray_valid[i] = True

    if len(edge_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32), ray_valid, np.zeros(0)

    return (np.array(edge_pts, dtype=np.float32),
            ray_valid,
            np.array(strengths, dtype=np.float32))


def fit_ellipse_ransac(
    points:     np.ndarray,
    n_iters:    int   = RANSAC_ITERS,
    thresh:     float = RANSAC_THRESH,
) -> Tuple[Optional[tuple], np.ndarray]:
    N = len(points)
    if N < 6:
        return None, np.zeros(N, dtype=bool)

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
        idx     = rng.choice(N, 5, replace=False)
        sample  = points[idx]

        try:
            ell = cv2.fitEllipse(sample)
        except cv2.error:
            continue

        (cx, cy), (ma, mi), angle = ell
        if ma < 1 or mi < 1 or ma > 200 or mi > 200:
            continue
        if ma / (mi + 1e-6) > 4.0:
            continue

        inliers = _ellipse_inliers(points, ell, thresh)
        n_in    = int(inliers.sum())

        if n_in > best_n_inliers:
            best_n_inliers = n_in
            best_inliers   = inliers
            best_ellipse   = ell

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
    (cx, cy), (ma, mi), angle_deg = ellipse
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy

    u =  dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a

    ra = ma / 2.0 + 1e-6
    rb = mi / 2.0 + 1e-6

    f    = (u / ra)**2 + (v / rb)**2
    dist = np.abs(f - 1.0) * (ra * rb) / (ra + rb)

    return dist < thresh


def estimate_pupil_seed(
    patch:          np.ndarray,
    limbus_ellipse: Optional[tuple],
    iris_center:    Tuple[float, float],
    iris_radius:    float,
) -> Tuple[Optional[Tuple[float,float]], float]:
    H, W = patch.shape[:2]

    mask = np.zeros((H, W), dtype=np.uint8)
    if limbus_ellipse is not None:
        cv2.ellipse(mask, limbus_ellipse, 255, -1)
    else:
        cv2.circle(mask, (int(iris_center[0]), int(iris_center[1])),
                   int(iris_radius * 0.9), 255, -1)

    if cv2.countNonZero(mask) < 5:
        return None, 0.0

    masked = cv2.bitwise_and(patch, patch, mask=mask)

    iris_pixels = patch[mask > 0]
    if len(iris_pixels) < 5:
        return None, 0.0

    dark_thresh = float(np.percentile(iris_pixels, 25))
    _, dark_mask = cv2.threshold(masked, int(dark_thresh), 255,
                                 cv2.THRESH_BINARY_INV)
    dark_mask = cv2.bitwise_and(dark_mask, mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k, iterations=1)

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


def compute_limbus_quality(
    ellipse:          Optional[tuple],
    inlier_mask:      np.ndarray,
    edge_points:      np.ndarray,
    ray_valid:        np.ndarray,
    iris_seed_radius: float,
) -> Tuple[float, float, float]:
    angular_coverage = float(ray_valid.sum()) / max(len(ray_valid), 1)

    if ellipse is None or inlier_mask.sum() < 5:
        return angular_coverage, 999.0, 0.0

    inlier_pts = edge_points[inlier_mask]
    dists      = _point_to_ellipse_dist(inlier_pts, ellipse)
    fit_resid  = float(np.mean(dists)) if len(dists) > 0 else 999.0

    (cx, cy), (ma, mi), _ = ellipse
    ratio = min(ma, mi) / (max(ma, mi) + 1e-6)
    size_ok = 0.5 < (ma / 2) / (iris_seed_radius + 1e-6) < 2.0

    coverage_score = angular_coverage
    residual_score = float(np.clip(1.0 - fit_resid / 5.0, 0.0, 1.0))
    shape_score    = ratio * (1.0 if size_ok else 0.3)

    quality = 0.4 * coverage_score + 0.4 * residual_score + 0.2 * shape_score
    return angular_coverage, fit_resid, float(np.clip(quality, 0.0, 1.0))


def _point_to_ellipse_dist(
    points: np.ndarray,
    ellipse: tuple,
) -> np.ndarray:
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


def step3_limbus(
    step2_result: "IlluminationResult",
    iris_radius_frame: float,
) -> LimbusResult:
    patch      = step2_result.blended
    iris_c     = step2_result.step1.iris_center
    warp_M     = step2_result.step1.warp_M

    scale      = float(np.linalg.norm(warp_M[0, :2]))
    iris_r_px  = float(np.clip(iris_radius_frame * scale, 3.0, min(PATCH_W, PATCH_H) / 2.0 - 1))

    profiles, r_coords, angles = sample_radial_rays(
        patch, iris_c, iris_r_px,
        n_rays=NUM_RAYS, band=RAY_SEARCH_BAND
    )

    edge_pts, ray_valid, strengths = extract_edge_points(
        profiles, r_coords, angles, iris_c, iris_r_px
    )

    if len(edge_pts) >= MIN_VALID_RAYS:
        ellipse, inlier_mask = fit_ellipse_ransac(edge_pts)
    else:
        ellipse      = None
        inlier_mask  = np.zeros(len(edge_pts), dtype=bool)

    if ellipse is None:
        cx, cy = iris_c
        ellipse = (
            (float(cx), float(cy)),
            (float(iris_r_px * 2), float(iris_r_px * 2)),
            0.0
        )
        inlier_mask = np.ones(len(edge_pts), dtype=bool)

    (ecx, ecy), (ma, mi), eangle = ellipse
    limbus_center = (float(ecx), float(ecy))
    limbus_axes   = (float(ma / 2.0), float(mi / 2.0))
    ecc           = float(np.sqrt(max(0, 1.0 - (min(ma,mi)/max(ma,mi,1e-6))**2)))

    pupil_center, pupil_radius = estimate_pupil_seed(
        patch, ellipse, iris_c, iris_r_px
    )

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


if __name__ == "__main__":
    import argparse
    print("preprocessing_pipeline.py loaded OK - this module is meant to be imported,")
    print("not run directly, by calibrate.py / run_session.py / main_webcam_pipeline.py.")
