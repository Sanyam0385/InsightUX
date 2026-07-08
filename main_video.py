import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import numpy as np

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
)
from inference_pipeline import InsightUXPipeline

pipeline  = InsightUXPipeline("models/gaze_cnn.onnx")
face_mesh = create_face_mesh(static_image_mode=False)

cap        = cv2.VideoCapture("test.mp4")
cam_matrix = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if cam_matrix is None:
        cam_matrix = estimate_camera_matrix(frame.shape)

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        continue

    lms = results.multi_face_landmarks[0].landmark

    head_pose = estimate_head_pose(lms, frame.shape, cam_matrix)
    if head_pose is None:
        continue

    # Step 1
    left_s1 = step1_normalize(
        frame, lms, head_pose,
        LEFT_EYE_INDICES, LEFT_EAR_INDICES, LEFT_IRIS_INDICES
    )

    # Step 2 - iris radius computed from landmarks, not hardcoded
    left_ir = compute_iris_radius(lms, LEFT_IRIS_INDICES, frame.shape)
    left_s2 = step2_illumination(left_s1, left_ir)

    head_pose_vec = np.array(
        [head_pose.pitch, head_pose.yaw, head_pose.roll],
        dtype=np.float32
    )

    gvx, gvy, pitch, yaw = pipeline.predict_gaze_vector(
        left_s2.blended,
        head_pose_vec
    )

    frame_count += 1
    print(f"Frame {frame_count} | Gaze: ({gvx:.3f}, {gvy:.3f})")

cap.release()
print(f"Done. Processed {frame_count} frames.")