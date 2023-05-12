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
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    
    A1 = np.column_stack((C1[0, 0]-C1[2, 0]*x1, C1[0, 1]-C1[2, 1]*x1, C1[0, 2]-C1[2, 2]*x1, C1[0, 3]-C1[2, 3]*x1))
    A2 = np.column_stack((C1[1, 0]-C1[2, 0]*y1, C1[1, 1]-C1[2, 1]*y1, C1[1, 2]-C1[2, 2]*y1, C1[1, 3]-C1[2, 3]*y1))
    A3 = np.column_stack((C2[0, 0]-C2[2, 0]*x2, C2[0, 1]-C2[2, 1]*x2, C2[0, 2]-C2[2, 2]*x2, C2[0, 3]-C2[2, 3]*x2))
    A4 = np.column_stack((C2[1, 0]-C2[2, 0]*y2, C2[1, 1]-C2[2, 1]*y2, C2[1, 2]-C2[2, 2]*y2, C2[1, 3]-C2[2, 3]*y2))

    # calculate the 3D coordinates for each point
    N = pts1.shape[0]
    W = np.zeros((N, 4))
    for ind in range(N):
        A = np.vstack((A1[ind, :], A2[ind, :], A3[ind, :], A4[ind, :]))
        _, _, vh = np.linalg.svd(A)
        W[ind, :] = vh[-1, :]

    # normalize homogeneous coordinates
    W = W / W[:, 3].reshape(-1, 1)

    # project to 2D points
    proj1 = C1 @ W.T
    proj2 = C2 @ W.T
    proj1 = proj1[:2, :] / proj1[2, :]
    proj2 = proj2[:2, :] / proj2[2, :]

    # compute error
    err = np.sum((proj1.T - pts1)**2 + (proj2.T - pts2)**2)

    return W[:, :3], err
def triangulate2(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1), dtype=float)))
    pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1),dtype=float)))
    
    def fun(p, C1, pts1_hom, C2, pts2_hom):
        P = p.reshape((-1, 3)).T
        err1 = C1 @ np.vstack((P, np.ones((1, P.shape[1]))))
        err2 = C2 @ np.vstack((P, np.ones((1, P.shape[1]))))
        denom1 = err1[2,:]
        denom1[np.isclose(denom1, 0, atol=1e3)] = 1e-3  # Replace zeros with a small value
        err1 /= denom1
        denom2 = err2[2,:]
        denom2[np.isclose(denom2, 0, atol=1e-3)] = 1e-3  # Replace zeros with a small value
        err2 /= denom2
        pts1_hom_3d = np.vstack((pts1_hom.T, np.ones((1, pts1_hom.shape[0]), dtype=float)))
        pts2_hom_3d = np.vstack((pts2_hom.T, np.ones((1, pts2_hom.shape[0]), dtype=float)))
        return np.concatenate((err1[:2,:] - pts1_hom_3d[:2,:], err2[:2,:] - pts2_hom_3d[:2,:]), axis=None)
    P0 = np.random.rand(pts1.shape[0], 3)
    P0[:,0] = P0[:,0] * (np.max(pts1[:,0]) - np.min(pts1[:,0])) + np.min(pts1[:,0])
    P0[:,1] = P0[:,1] * (np.max(pts1[:,1]) - np.min(pts1[:,1])) + np.min(pts1[:,1])
    P0[:,2] = P0[:,2]
    res = least_squares(fun, P0.ravel(), args=(C1, pts1_hom, C2, pts2_hom), loss='huber')
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
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # set the size of the window
    x1, y1 = int(round(x1)), int(round(y1))
    window_size = 11
    center = window_size//2
    sigma = 5
    search_range = 40

    # create gaussian weight matrix
    mask = np.ones((window_size, window_size))*center
    mask = np.repeat(np.array([range(window_size)]), window_size, axis=0) - mask
    mask = np.sqrt(mask**2+np.transpose(mask)**2)
    weight = np.exp(-0.5*(mask**2)/(sigma**2))
    weight /= np.sum(weight)

    if len(im1.shape) > 2:
        weight = np.repeat(np.expand_dims(weight, axis=2), im1.shape[-1], axis=2)

    # get the epipolar line
    p = np.array([[x1], [y1], [1]])
    l2 = np.dot(F, p)

    # get the patch around the pixel in image1
    patch1 = im1[y1-center:y1+center+1, x1-center:x1+center+1]
    # get the points on the epipolar line
    h, w, _ = im2.shape
    Y = np.array(range(y1-search_range, y1+search_range))
    X = np.round(-(l2[1]*Y+l2[2])/l2[0]).astype(np.int)
    valid = (X >= center) & (X < w - center) & (Y >= center) & (Y < h - center)
    X, Y = X[valid], Y[valid]

    min_dist = None
    x2, y2 = None, None
    for i in range(len(X)):
        # get the patch around the pixel in image2
        patch2 = im2[Y[i]-center:Y[i]+center+1, X[i]-center:X[i]+center+1]
        # calculate the distance
        dist = np.sum((patch1-patch2)**2*weight)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            x2, y2 = X[i], Y[i]

    return x2, y2

def epipolarCorrespondence2(im1, im2, f, x1, y1):
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
    import cv2
    # Convert images to grayscale
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Use ORB feature detector and descriptor
    orb = cv2.ORB_create()

    # Use Flann-based matcher
    flann = cv2.FlannBasedMatcher()

    # Detect and match features
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Convert keypoint coordinates to numpy arrays
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate the fundamental matrix
    F, _ = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_8POINT)

    # Estimate the essential matrix
    E = np.dot(np.transpose(k2), np.dot(F, k1))

    # Perform SVD on the essential matrix
    U, _, Vt = np.linalg.svd(E)

    # Ensure that the singular values are consistent
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Compute the rotation matrix and translation vector
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = U @ W @ Vt
    t = U[:, 2]

    return [R, t]


'''
Q3.2: Implement a monocular visual odometry.
    Input:  datafolder, the folder of the provided monocular video sequence
            GT_pose, the provided ground-truth (GT) pose for each frame
            plot=True, draw the estimated and the GT camera trajectories in the same plot
    Output: trajectory, the estimated camera trajectory (with scale aligned)        

'''
def visualOdometry(datafolder, GT_Pose, plot=True):
    import os
    import cv2
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Replace pass by your implementation
    file=open('../data/Intrinsic4Recon.npz','rb')
    file=file.readlines()
    d={}
    for i in file:
        i=i.decode('utf-8')
        i=i.split(':')
        d[i[0]]=np.array(list(i[1].split()),dtype=float)

    K1 = d['K1'].reshape(3,3)
    K2 = d['K2'].reshape(3,3)
    # Initialize the camera trajectory
    trajectory = np.zeros((len(os.listdir(datafolder)), 3))
    # Initialize the first camera pose as [I|0]
    pose = np.eye(4)

    # Loop over all image frames in the video sequence
    filelist=os.listdir(datafolder)
    for i in range(len(filelist)-1):
        # Load the current image frame
        im = cv2.imread(os.path.join(datafolder, str(filelist[i])))

        # Estimate the relative camera motion from the previous frame
        if i > 0:
            # Load the previous image frame
            prev_im = cv2.imread(os.path.join(datafolder, str(filelist[i-1])))

            # Estimate the fundamental matrix and essential matrix
            [R, t] = essentialDecomposition(prev_im, im, K1, K2)
            t_hat = np.array([[0, -t[2], t[1]],
                      [t[2], 0, -t[0]],
                      [-t[1], t[0], 0]])

            # Compute the Essential matrix
            E = np.dot(t_hat, R)
            E = np.dot(np.transpose(K1), np.dot(E, K2))

              # Scale the translation using the ground-truth data
            scale = helper.getAbsoluteScale(GT_Pose[i - 1], GT_Pose[i])
            t = scale * t

            # Update the camera pose using the relative rotation and scaled translation
            pose[:3, :3] = np.dot(R, pose[:3, :3])
            pose[:3, 3] = pose[:3, 3] + np.dot(pose[:3, :3], t)

            # Compute the absolute camera pose using the scaled translation
            abs_t = np.dot(pose[:3, :3], t)
            abs_pose = np.eye(4)
            abs_pose[:3, :3] = pose[:3, :3]
            abs_pose[:3, 3] = trajectory[i - 1] + abs_t

            # Save the absolute camera position in the trajectory
            trajectory[i, :] = abs_pose[:3, 3]
    print(trajectory.shape)
    print(trajectory)
    # Plot the camera trajectory if requested
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the estimated trajectory in blue
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b')

        # Plot the ground-truth trajectory in red
        ax.plot(GT_Pose[:, 3], GT_Pose[:, 7], GT_Pose[:, 11], 'r')

        # Set the plot title and labels
        ax.set_title('Camera Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Show the plot
        plt.show()

    return trajectory
    pass



