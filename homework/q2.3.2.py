import cv2
import numpy as np
from helper import *
from submission import *
img1 = cv2.imread("../data/image1.jpg")
img2 = cv2.imread("../data/image2.jpg")

data = np.load('q2.3_1.npz')
pts1 = data['src_points'] if data['src_points'].shape[0] < 500 else data['src_points'][0:500]
pts2 = data['dst_points'] if data['dst_points'].shape[0] < 500 else data['dst_points'][0:500]
print(img1.shape)
M = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
F = eightpoint(pts1, pts2, M)
print(F)
np.savez('q2.3_2.npz', F=F, M=M)

displayEpipolarF(img1, img2, F)
