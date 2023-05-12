import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform
img1 = cv2.imread("../data/image1.jpg")
img2 = cv2.imread("../data/image2.jpg")

    
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create()


keypoints1, descriptors1 = sift.detectAndCompute(gray_img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_img2, None)


bf_matcher = cv2.FlannBasedMatcher()


matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for match1, match2 in matches:
    if match1.distance < 0.8 * match2.distance:
        good_matches.append(match1)
good_matches.sort(key=lambda x: x.distance)

src_points = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

src_points=src_points.reshape((-1, 2))
dst_points=dst_points.reshape((-1, 2))

_, inliers = ransac((src_points, dst_points), AffineTransform,
                     min_samples=4, residual_threshold=8, max_trials=10000)


src_points = src_points[inliers]
dst_points = dst_points[inliers]

    
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

src_pts=src_points.squeeze()
dst_pts=dst_points.squeeze()


np.savez('q2.3_1.npz', src_points=src_pts, dst_points=dst_pts)


cv2.imwrite('2_3_1_fig1.png', img_matches)

#cv2.imshow('Keypoints and Matches', img_matches)
#cv2.waitKey(0)
#cv2.destroyAllWindows()