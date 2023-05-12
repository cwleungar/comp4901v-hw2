import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from helper import *
from submission import *
# Load the data
file=open('../data/VisPts.npz','r')
file=file.readlines()
xys=[]
for i in file:
    i=i.split('   ')
    xys.append([int(i[0]),int(i[1])])

M1 = np.hstack((np.identity(3), np.zeros((3, 1))))
M2=np.load('q2.4_3.npz')['M2']
C2=np.load('q2.4_3.npz')['C2']
x1=np.array([i[0] for i in xys])
y1=np.array([i[1] for i in xys])
file=open('../data/Intrinsic4Recon.npz','rb')
file=file.readlines()
d={}
for i in file:
    i=i.decode('utf-8')
    i=i.split(':')
    d[i[0]]=np.array(list(i[1].split()),dtype=float)

K1 = d['K1'].reshape(3,3)
K2 = d['K2'].reshape(3,3)

F=np.load('q2.5_1.npz')['F']
im1 = cv2.imread("../data/image1.jpg")
im2 = cv2.imread("../data/image2.jpg")

M = max(im1.shape[0], im1.shape[1], im2.shape[0], im2.shape[1])
pts1 = np.column_stack((x1, y1))
pts2 = np.array([epipolarCorrespondence(im1, im2, F, x, y) for x, y in pts1])


pts_3d, err  =  triangulate(K1@M1, pts1, C2, pts2)
print(pts_3d)
positive_depth_mask = pts_3d[:, 2] > 0 
pts_3d_positive = pts_3d[positive_depth_mask]


# Plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts_3d_positive[:, 0]/np.max(pts_3d_positive[:, 0]), pts_3d_positive[:, 1]/np.max(pts_3d_positive[:, 1]), pts_3d_positive[:, 2]/np.max(pts_3d_positive[:, 2]), c='b', marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Plot the corresponding image points on img1
fig2, ax2 = plt.subplots()
ax2.imshow(im1)
ax2.scatter(pts1[:, 0], pts1[:, 1], c='r', marker='o')
ax2.set_title('Image 1 with Corresponding Points')
plt.show()
np.savez('q2.5_2.npz', F=F, C1=K1@M1,C2=C2)
