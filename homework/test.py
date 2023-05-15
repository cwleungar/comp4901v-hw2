import numpy as np
from helper import *
from submission import *
import cv2
# Define the camera intrinsics
K1 = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])
K2 = K1
im1= cv2.imread("../data/image1.jpg")
im2=im1
GT_Pose_flat = np.random.rand(10, 12)  # 10 poses, each represented as a flattened 3x4 camera matrix
GT_Pose = GT_Pose_flat.reshape(-1, 3, 4)

# Define the expected results
expected_R = GT_Pose[1][:3, :3] @ np.linalg.inv(GT_Pose[0][:3, :3])
expected_t = GT_Pose[1][:3, 3] - expected_R @ GT_Pose[0][:3, 3]
expected_scale = np.linalg.norm(GT_Pose[1][:3, 3] - GT_Pose[0][:3, 3])

# Compute the relative camera motion between the frames
[R1, R2], [t1, t2] = essentialDecomposition(im1, im2, K1, K2)

# Compute the scale factor between the ground-truth poses
scale = getAbsoluteScale(GT_Pose[0], GT_Pose[1])

# Print the results
print("R1: ", R1)
print("R2: ", R2)
print("expected_R: ", expected_R)
assert np.allclose(R2, expected_R)
assert np.allclose(t2, expected_t)
assert np.isclose(scale, expected_scale)