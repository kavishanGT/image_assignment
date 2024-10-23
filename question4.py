import cv2 as cv
import numpy as np

# Load the images
img1 = cv.imread('images/graf/img1.ppm')
img5 = cv.imread('images/graf/img5.ppm')

# Convert the images to grayscale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray5 = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv.SIFT_create()

# Detect SIFT features and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints5, descriptors5 = sift.detectAndCompute(gray5, None)

# Match features using FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors5, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Draw matches (for visualization purposes)
img_matches = cv.drawMatches(img1, keypoints1, img5, keypoints5, good_matches, None)
cv.imshow('Matches', img_matches)
cv.waitKey(0)
cv.destroyAllWindows()

# Extract matched keypoints
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts5 = np.float32([keypoints5[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography using RANSAC
H, mask = cv.findHomography(pts1, pts5, cv.RANSAC, 5.0)

# Display the homography matrix
print("Homography Matrix:\n", H)

# Get the dimensions of img5
h5, w5 = img5.shape[:2]

# Warp img1 using the homography matrix H
img1_warped = cv.warpPerspective(img1, H, (w5, h5))

# Stitch the images together by blending
result = img5.copy()
result[np.where(img1_warped > 0)] = img1_warped[np.where(img1_warped > 0)]



# Display the results
cv.imshow('Warped Image', img1_warped)
cv.imshow('Stitched Image', result)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the final stitched image
cv.imwrite('stitched_image.png', result)
