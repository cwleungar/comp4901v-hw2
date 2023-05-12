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
img2 = cv2.imread("../data/image2.jpg")
data = np.load('q2.3_1.npz')
pts1 = data['src_points'] if data['src_points'].shape[0] < 300 else data['src_points'][0:300]
pts2 = data['dst_points'] if data['dst_points'].shape[0] < 300 else data['dst_points'][0:300]
file=open('../data/Intrinsic4Recon.npz','rb')
file=file.readlines()
d={}
for i in file:
    i=i.decode('utf-8')
    i=i.split(':')
    d[i[0]]=np.array(list(i[1].split()),dtype=float)

K1 = d['K1'].reshape(3,3)
K2 = d['K2'].reshape(3,3)
F=np.load('q2.3_2.npz')['F']
E = essentialMatrix(F, K1, K2)

M1 = np.hstack((np.identity(3), np.zeros((3, 1))))
M2s = camera2(E)
best_M2 = None
max_num_in_front = 0

# Loop through the four possible M2 matrices

for i in range(4):
    M2=M2s[:,:,i]
    P, err = triangulate(K1@M1, pts1, K2@M2, pts2)

    # Check if the projected points have positive depth
    P_hom = np.hstack((P, np.ones((P.shape[0], 1))))
    pts1_proj = K1@M1 @ P_hom.T
    pts2_proj = K2@M2 @ P_hom.T
    z1 = pts1_proj[2, :]
    z2 = pts2_proj[2, :]
    in_front = (z1 > 0) & (z2 > 0)
    num_in_front = np.sum(in_front)
    print(num_in_front)
    # Update the best M2 matrix and the maximum number of points in front if necessary
    if num_in_front > max_num_in_front:
        best_M2 = (M2,K2@M2,P)
        max_num_in_front = num_in_front
# Save the best M2 matrix, C2, and P
np.savez('q2.4_3.npz', M2=best_M2[0], C2=best_M2[1], P=best_M2[2])