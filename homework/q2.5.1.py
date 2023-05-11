import cv2
import numpy as np
from helper import *
from submission import *
img1 = cv2.imread("../data/image1.jpg")
img2 = cv2.imread("../data/image2.jpg")

data = np.load('q2.3_1.npz')
pts1 = data['src_points'][0:100]
pts2 = data['dst_points'][0:100]
M=max(img1.shape[0],img1.shape[0])
F = eightpoint(pts1, pts2, M)
np.savez('q2.5_1.npz', pts1=pts1, pts2=pts2,F=F)

epipolarMatchGUI(img1, img2, F)
