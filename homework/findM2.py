import numpy as np
from helper import *
from skimage.measure import ransac
from skimage.transform import AffineTransform
from helper import _epipoles

from submission import eightpoint, essentialMatrix, triangulate
'''
Q2.4.3:
    1. Load point correspondences calculated and saved in Q2.3.1
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

# Initialize variables to keep track of the best M2 matrix and the number of points it triangulates
data = np.load('q2.3_1.npz')
pts1 = data['src_points']
pts2 = data['dst_points']
K1 = np.load('../data/intrinsics.npz')['K1']
K2 = np.load('../data/intrinsics.npz')['K2']

F = eightpoint(pts1, pts2, 256)
E = essentialMatrix(F, K1, K2)
M1 = np.hstack((np.identity(3), np.zeros((3, 1))))
M2s = camera2(E)
best_M2 = None
max_num_in_front = 0

# Loop through the four possible M2 matrices
for M2 in M2s:
    
    P, err = triangulate(K1@M1, pts1, K2@M2, pts2)
    
    # Count the number of 3D points that are in front of both cameras
    num_in_front = np.sum((P[:, 2] > 0) & (P[:, 5] > 0))
    
    # Update the best M2 matrix and the maximum number of points in front if necessary
    if num_in_front > max_num_in_front:
        best_M2 = (M2,K2@M2,P)
        max_num_in_front = num_in_front

# Save the best M2 matrix, C2, and P
np.savez('q3_3.npz', M2=best_M2[0], C2=best_M2[1], P=best_M2[2])