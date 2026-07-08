import cv2
import numpy as np
import os
from preprocessing.preprocessing_pipeline import (
    create_face_mesh, estimate_camera_matrix,
    estimate_head_pose, LEFT_EYE_INDICES,
    LEFT_EAR_INDICES, LEFT_IRIS_INDICES,
    compute_iris_radius, step1_normalize,
    step2_illumination
)
from inference_pipeline import InsightUXPipeline

pipeline  = InsightUXPipeline("models/gaze_cnn.onnx")
face_mesh = create_face_mesh(static_image_mode=True)

# Put all 9 images in a folder called 'test_images'
# Name them: 1_topleft.jpg, 2_topcenter.jpg, 3_topright.jpg etc.
# OR just name them 1.jpg, 2.jpg ... 9.jpg in order

image_dir = "test_images"
positions = [
    "top-left", "top-center", "top-right",
    "mid-left", "center",     "mid-right",
    "bot-left", "bot-center", "bot-right"
]

files = sorted([f for f in os.listdir(image_dir)
                if f.lower().endswith(('.jpg','.png','.jpeg'))])

print(f"{'File':<25} {'Position':<15} {'H.Pitch':>8} {'H.Yaw':>8} {'CNN.Pitch':>10} {'CNN.Yaw':>10}")
print("-" * 80)

for i, fname in enumerate(files[:9]):
    pos   = positions[i] if i < len(positions) else f"pos{i+1}"
    frame = cv2.imread(os.path.join(image_dir, fname))
    if frame is None:
        print(f"{fname:<25} Could not read")
        continue

    cam_matrix = estimate_camera_matrix(frame.shape)
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results    = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print(f"{fname:<25} No face detected")
        continue

    lms       = results.multi_face_landmarks[0].landmark
    head_pose = estimate_head_pose(lms, frame.shape, cam_matrix)
    if head_pose is None:
        print(f"{fname:<25} Head pose failed")
        continue

    left_s1 = step1_normalize(frame, lms, head_pose,
                              LEFT_EYE_INDICES, LEFT_EAR_INDICES,
                              LEFT_IRIS_INDICES)
    left_ir = compute_iris_radius(lms, LEFT_IRIS_INDICES, frame.shape)
    left_s2 = step2_illumination(left_s1, left_ir)

    head_pose_vec = np.array(
        [head_pose.pitch, head_pose.yaw, head_pose.roll],
        dtype=np.float32
    )
    _, _, pitch, yaw = pipeline.predict_gaze_vector(
        left_s2.blended, head_pose_vec
    )

    print(f"{fname:<25} {pos:<15} "
          f"{head_pose.pitch:>8.2f} {head_pose.yaw:>8.2f} "
          f"{pitch:>10.4f} {yaw:>10.4f}")