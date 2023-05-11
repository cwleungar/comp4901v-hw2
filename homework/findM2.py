import numpy as np
from helper import *
from skimage.measure import ransac
from skimage.transform import AffineTransform
from helper import _epipoles
import cv2
from submission import eightpoint, essentialMatrix, triangulate
'''
Q2.4.3:
    1. Load point correspondences calculated and saved in Q2.3.1
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

# Initialize variables to keep track of the best M2 matrix and the number of points it triangulates
img1 = cv2.imread("../data/image1.jpg")

data = np.load('q2.3_1.npz')
pts1 = data['src_points'][:100]
pts2 = data['dst_points'][:100]
file=open('../data/Intrinsic4Recon.npz','rb')
file=file.readlines()
d={}
for i in file:
    i=i.decode('utf-8')
    i=i.split(':')
    d[i[0]]=np.array(list(i[1].split()),dtype=float)

K1 = d['K1'].reshape(3,3)
K2 = d['K2'].reshape(3,3)
M=max(img1.shape[0],img1.shape[0])

F = eightpoint(pts1, pts2, M)
E = essentialMatrix(F, K1, K2)

M1 = np.hstack((np.identity(3), np.zeros((3, 1))))
M2s = camera2(E)
best_M2 = None
max_num_in_front = 0

# Loop through the four possible M2 matrices
for M2 in M2s:
    K2_ext = np.hstack((K2, np.zeros((3, 1))))

    P, err = triangulate(K1@M1, pts1, K2_ext@M2, pts2)
    
    # Count the number of 3D points that are in front of both cameras
    num_in_front = np.sum(P[:, 2] > 0)
    
    # Update the best M2 matrix and the maximum number of points in front if necessary
    if num_in_front > max_num_in_front:
        best_M2 = (M2,K2_ext@M2,P)
        max_num_in_front = num_in_front

# Save the best M2 matrix, C2, and P
np.savez('q2.4_3.npz', M2=best_M2[0], C2=best_M2[1], P=best_M2[2])