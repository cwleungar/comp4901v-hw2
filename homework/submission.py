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
    # Scale the point coordinates by M
    pts1 = pts1 / M
    pts2 = pts2 / M

    # Construct the A matrix
    N = pts1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x, y = pts1[i]
        u, v = pts2[i]
        A[i] = [x*u, x*v, x, y*u, y*v, y, u, v, 1]

    # Solve the homogeneous linear system using SVD
    _, _, V = svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce the singularity constraint on F
    U, s, Vt = svd(F)
    s[2] = 0
    F = U @ np.diag(s) @ Vt

    # Refine the solution using local minimization
    F = helper.refineF(F, pts1, pts2)

    # Unscale the fundamental matrix
    T = np.diag([1/M, 1/M, 1])
    F = T.T @ F @ T

    return F
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
        P = p.reshape((-1, 3)).T
        err1 = C1 @ np.vstack((P, np.ones((1, P.shape[1]))))
        err2 = C2 @ np.vstack((P, np.ones((1, P.shape[1]))))
        denom1 = err1[2,:]
        denom1[np.isclose(denom1, 0, atol=1e-15)] = 1e-15  # Replace zeros with a small value
        err1 /= denom1
        denom2 = err2[2,:]
        denom2[np.isclose(denom2, 0, atol=1e-15)] = 1e-15  # Replace zeros with a small value
        err2 /= denom2
        pts1_hom_3d = np.vstack((pts1_hom.T, np.ones((1, pts1_hom.shape[0]))))
        pts2_hom_3d = np.vstack((pts2_hom.T, np.ones((1, pts2_hom.shape[0]))))
        return np.concatenate((err1[:2,:] - pts1_hom_3d[:2,:], err2[:2,:] - pts2_hom_3d[:2,:]), axis=None)
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
    def find_matching_point(im1, im2, x1, y1, F, window_size=5, sigma=1.0):
        # Compute the window around the point (x1, y1) in im1
        half_window_size = window_size // 2
        x1_min = max(x1 - half_window_size, 0)
        x1_max = min(x1 + half_window_size + 1, im1.shape[1])
        y1_min = max(y1 - half_window_size, 0)
        y1_max = min(y1 + half_window_size + 1, im1.shape[0])
        window1 = im1[y1_min:y1_max, x1_min:x1_max, :]

        # Compute the Gaussian kernel for weighting the window
        kernel = np.zeros((window_size, window_size))
        center = (window_size - 1) / 2
        for i in range(window_size):
            for j in range(window_size):
                kernel[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))

        # Normalize the kernel
        kernel /= np.sum(kernel)

        # Find the best match in im2
        best_match_distance = np.inf
        best_match_x = None
        best_match_y = None
        for x2, y2 in zip(x_range, y_range):
            # Compute the window around the point (x2, y2) in im2
            x2_min = max(int(x2) - half_window_size, 0)
            x2_max = min(int(x2) + half_window_size + 1, im2.shape[1])
            y2_min = max(int(y2) - half_window_size, 0)
            y2_max = min(int(y2) + half_window_size + 1, im2.shape[0])
            window2 = im2[y2_min:y2_max, x2_min:x2_max, :]
            output_shape = (5, 5, 3)
            pad_along_height = max((output_shape[0] - window2.shape[0]), 0)
            pad_along_width = max((output_shape[1] - window2.shape[1]), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            
            # Pad the image along each dimension
            window2_padded = np.pad(window2, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
            # Apply the Gaussian weighting to the windows
            window1_weighted = np.zeros_like(window1)
            window2_weighted = np.zeros_like(window2_padded)
            for i in range(3):
                window1_weighted[..., i] = window1[..., i] * kernel
                window2_weighted[..., i] = window2_padded[..., i] * kernel

            # Compute the distance between the windows
            distance = np.sum((window1_weighted - window2_weighted)**2)

            # Update the best match if the distance is smaller
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_x = x2
                best_match_y = y2

        return best_match_x, best_match_y
    # Replace pass by your implementation
        # Compute the epipolar line in im2 corresponding to the point (x1, y1) in im1
    epipolar_line = f @ np.array([x1, y1, 1])
    a, b, c = epipolar_line
    
    # Compute the range of possible y-coordinates in im2 for the given x-coordinate
    y_range = np.arange(im2.shape[0])
    x_range = (-c - b*y_range) / a
    valid_mask = (x_range >= 0) & (x_range < im2.shape[1])
    x_range = x_range[valid_mask]
    y_range = y_range[valid_mask]
    
    # Find the best match in im2 using the find_matching_point function
    x2, y2 = find_matching_point(im1, im2, x1, y1, f)
    
    return x2, y2
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



