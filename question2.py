import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the architectural image and flag image
architectural_image = cv.imread('images/005.jpg')
flag_image = cv.imread('images/flag.png')
gray_image = cv.cvtColor(architectural_image, cv.COLOR_BGR2GRAY)

# Resize flag image to a smaller size for easier processing (optional)
flag_image = cv.resize(flag_image, (300, 200))

# Select 4 points on the architectural image where the flag will be mapped
# These points should form a quadrilateral on the planar surface
pts_src = np.array([[217, 156], [467, 302], [465, 537], [194, 526]], dtype=float)  # example points


# Define the 4 corner points of the flag image
height, width = flag_image.shape[:2]
pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)

# Compute the homography matrix
H, status = cv.findHomography(pts_dst, pts_src)
print(H)

# Warp the flag image using the homography matrix
warped_flag = cv.warpPerspective(flag_image, H, (architectural_image.shape[1], architectural_image.shape[0]))

# Create a mask of the warped flag to use for blending
flag_mask = np.zeros_like(architectural_image, dtype=np.uint8)
flag_mask = cv.fillConvexPoly(flag_mask, pts_src.astype(int), (10, 10, 10))

# Invert the mask to black out the area where the flag will be placed on the architectural image
inv_flag_mask = cv.bitwise_not(flag_mask)

# Black out the region in the architectural image where the flag will be placed
arch_background = cv.bitwise_and(architectural_image, inv_flag_mask)

# Blend the warped flag with the architectural image
result = cv.add(arch_background, warped_flag)


# Display the result
#plt.figure(figsize=(10, 8))
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title('Architectural Image with Superimposed Flag')
plt.axis('off')

#plt.imshow(cv.cvtColor(architectural_image, cv.COLOR_BGR2RGB))
plt.show()
