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
    #F = helper.refineF(F, pts1, pts2)

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
    from skimage.measure import ransac
    from skimage.transform import AffineTransform
    # Convert images to grayscale
    gray_img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()


    keypoints1, descriptors1 = sift.detectAndCompute(gray_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_img2, None)


    bf_matcher = cv2.FlannBasedMatcher()


    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=4)

    good_matches = []
    for match_list in matches:
        if len(match_list) < 2:
            continue
        match1, match2 = match_list[:2]
        good_matches.append(match1)

    good_matches.sort(key=lambda x: x.distance)

    src_points = np.float32([keypoints1[match.queryIdx].pt for match in good_matches[:150]]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in good_matches[:150]]).reshape(-1, 1, 2)

    src_points=src_points.reshape((-1, 2))
    dst_points=dst_points.reshape((-1, 2))
    
    src_pts=src_points.squeeze()
    dst_pts=dst_points.squeeze()
    _, inliers = ransac((src_points, dst_points), AffineTransform,
                         min_samples=4, residual_threshold=8, max_trials=10000)
    src_points = src_points[inliers]
    dst_points = dst_points[inliers]
    M=max(im1.shape[0],im1.shape[1],im2.shape[0],im2.shape[1])
    
    F= eightpoint(src_pts, dst_pts, M)
    E = essentialMatrix(F, k1,k2)

    # Compute the decomposition of the essential matrix
    U, _, Vt = svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 *= -1
        t1 *= -1
    if np.linalg.det(R2) < 0:
        R2 *= -1
        t2 *= -1
    def formT(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    M2s=[(R1,t1,formT(R1,t1)),(R1,t2,formT(R1,t2)),(R2,t1,formT(R2,t1)),(R2,t2,formT(R2,t2))]
    M1=np.hstack((np.eye(3), np.zeros((3, 1))))
    best_M2=None
    max_num_in_front=0
    for i in range(4):
        r,t,M2=M2s[i]
        t.reshape((3,1))
        M2c=M2[:3, :]
        P, err = triangulate(k1@M1, src_points, k2@M2c, dst_points)
        # Check if the projected points have positive depth
        P_hom = np.hstack((P, np.ones((P.shape[0], 1))))
        pts1_proj = k1@M1 @ P_hom.T
        pts2_proj = k2@M2c @ P_hom.T
        z1 = pts1_proj[2, :]
        z2 = pts2_proj[2, :]
        in_front = (z1 > 0) & (z2 > 0)
        num_in_front = np.sum(in_front)
        # Update the best M2 matrix and the maximum number of points in front if necessary
        if num_in_front > max_num_in_front:
            best_M2 = (r,t)
            max_num_in_front = num_in_front
    # Compute the camera poses for the four possible solutions
    return best_M2



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
    trajectory = np.zeros((len(os.listdir(datafolder)), 3),dtype= float)
    # Initialize the first camera pose as [I|0]
    pose=np.hstack((np.eye(3),np.zeros((3,1))))
    last_R=np.eye(3)
    last_t=np.zeros((3,1))
    # Loop over all image frames in the video sequence
    filelist=os.listdir(datafolder)
    for i in range(len(filelist)-1):
        # Load the current image frame
        im = cv2.imread(os.path.join(datafolder, str(filelist[i])))
        filelist.sort()
        # Estimate the relative camera motion from the previous frame
        if i==0:
            trajectory[i]=GT_Pose[i].reshape((3,4))[:,3]
        if i > 0:
            prev_im = cv2.imread(os.path.join(datafolder, str(filelist[i-1])))
            #[R1,R2],[t1,t2] = essentialDecomposition(prev_im, im, K1, K2)
            R,t = essentialDecomposition(prev_im, im, K1, K2)
            #poses = [np.hstack([R, t.reshape(3, 1)]) for R, t in [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]]
            # Select the camera pose that is most consistent with the ground truth pose

            # R and t are the relative rotation and translation between the current and the previous frame
            # We need to compute the absolute pose of the current frame
            # ti = ti−1 + scale ∗ (R i-1−>i ∗ t i-1−>i)
            tGT1=GT_Pose[i-1].reshape((3,4))[:,3]
            tGT2=GT_Pose[i].reshape((3,4))[:,3]
            scale = helper.getAbsoluteScale(tGT1, tGT2)
            #cur_t = cur_t + scale*cur_R@t.reshape((3, 1))
            #cur_R = R@cur_R
#
            RGT1=GT_Pose[i-1].reshape((3,4))[:,:3]
            RGT2=GT_Pose[i].reshape((3,4))[:,:3]
#
            #R_rel=RGT2.T@RGT1
            #t_rel = tGT2 - tGT1
            #P_prev = np.hstack((last_R, trajectory[i-1].reshape((3,1))))
            #P_curr = np.hstack((R, t.reshape((3,1))))
            #P_prev =GT_Pose[i-1].reshape((3,4))
            # Compute the relative rotation and translation between the two frames
            R_rel = R @ last_R.T #R @ np.linalg.inv(P_prev[:, :3])
            #t_rel = t.reshape((3, 1))
            #t_rel = np.linalg.inv(P_prev[:, :3]) @ (t_rel - P_prev[:, 3].reshape((3, 1)))
            t=t.reshape((3,1))
            t_rel = last_t + last_R.T @ (t - last_t)
            ## Compute the absolute translation vector of the current frame
            rti = scale * (R_rel @ t_rel.reshape((3,1)))
            trajectory[i] = trajectory[i-1] + rti.reshape((3,))
            last_R = R
            last_t = t
        if i%20==0:
            print(i,trajectory[i])
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



