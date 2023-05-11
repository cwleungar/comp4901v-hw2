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
file=open('../data/Intrinsic4Recon.npz','rb')
file=file.readlines()
d={}
for i in file:
    i=i.decode('utf-8')
    i=i.split(':')
    d[i[0]]=np.array(list(i[1].split()),dtype=float)

K1 = d['K1'].reshape(3,3)
K2 = d['K2'].reshape(3,3)
E=essentialMatrix(F, K1, K2)
epipolarMatchGUI(img1, img2, F)
print(E)