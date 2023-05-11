"""
Homework2.
Replace 'pass' by your implementation.
"""
import numpy as np
from scipy.linalg import svd
import helper
from scipy.optimize import least_squares

# Insert your package here

'''
Q2.3.2: Eight Point algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: f, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation

    pts1 = pts1 / M
    pts2 = pts2 / M
    
    a1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    a2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    a = np.kron(a1, a2)
    a = a.reshape((a1.shape[0], a2.shape[0], 9))
    a = a.transpose((2, 0, 1)).reshape((9, -1)).T
    _, _, v = svd(a)
    
    
    f = v[-1,:].reshape((3,3))
    u, s, v = svd(f)
    s[2] = 0
    f = u @ np.diag(s) @ v
    f = helper.refineF(f, pts1, pts2)
    t = np.diag([1/M, 1/M, 1])
    f = t.T @ f @ t
    
    return f
    pass


'''
Q2.4.1: Compute the essential matrix E.
    Input:  f, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(f, K1, K2):
    # Replace pass by your implementation
    e = K2.T @ f @ K1
    u, s, v = np.linalg.svd(e)
    s[2] = 0
    e = u @ np.diag(s) @ v.T
    return e
    pass

'''
Q2.4.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
    def fun(p, C1, pts1_hom, C2, pts2_hom):
        P = p.reshape((3,1))
        err1 = C1 @ np.vstack((P, 1))
        err2 = C2 @ np.vstack((P, 1))
        return np.concatenate((err1[:2,:] / err1[2,:] - pts1_hom.T, err2[:2,:] / err2[2,:] - pts2_hom.T), axis=None)
    
    P0 = np.zeros((pts1.shape[0], 3))
    
    res = least_squares(fun, P0.ravel(), args=(C1, pts1_hom, C2, pts2_hom), method='lm')
    P = res.x.reshape((-1, 3))
    
    pts1_proj = C1 @ np.vstack((P.T, np.ones((1, P.shape[0]))))
    pts2_proj = C2 @ np.vstack((P.T, np.ones((1, P.shape[0]))))
    err1 = np.linalg.norm(pts1_hom[:,:2] - pts1_proj[:2,:].T, axis=1)
    err2 = np.linalg.norm(pts2_hom[:,:2] - pts2_proj[:2,:].T, axis=1)
    err = np.sum(err1**2 + err2**2)
    
    return P, err
    pass


'''
Q2.5.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            f, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, f, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q3.1: Decomposition of the essential matrix to rotation and translation.
    Input:  im1, the first image
            im2, the second image
            k1, camera intrinsic matrix of the first frame
            k1, camera intrinsic matrix of the second frame
    Output: R, rotation
            r, translation

'''
def essentialDecomposition(im1, im2, k1, k2):
    # Replace pass by your implementation
    pass


'''
Q3.2: Implement a monocular visual odometry.
    Input:  datafolder, the folder of the provided monocular video sequence
            GT_pose, the provided ground-truth (GT) pose for each frame
            plot=True, draw the estimated and the GT camera trajectories in the same plot
    Output: trajectory, the estimated camera trajectory (with scale aligned)        

'''
def visualOdometry(datafolder, GT_Pose, plot=True):
    # Replace pass by your implementation
    pass



