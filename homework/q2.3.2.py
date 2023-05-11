import cv2
import numpy as np
from helper import *
from submission import *
img1 = cv2.imread("../data/image1.jpg")
img2 = cv2.imread("../data/image2.jpg")

data = np.load('q2.3_1.npz')
pts1 = data['src_points']
pts2 = data['dst_points']


F = eightpoint(pts1, pts2, 256)
print("calcuated F: ", F)
displayEpipolarF(img1, img2, F)
