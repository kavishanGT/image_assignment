import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image
im = cv.imread('images/the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)
gray_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Parameters
min_sigma = 2  # Minimum sigma for Gaussian
max_sigma = 10  # Maximum sigma
num_scales = 10  # Number of scales (i.e., different sigma values)
threshold = 0.03  # Threshold to filter weak responses

# Generate scale space for blob detection
scale_space = np.zeros((gray_image.shape[0], gray_image.shape[1], num_scales))

sigma_values = np.linspace(min_sigma, max_sigma, num_scales)

for i, sigma in enumerate(sigma_values):
    # Apply Gaussian Blur to smooth the image at this scale
    smoothed_image = cv.GaussianBlur(gray_image, (0, 0), sigma)
    
    # Compute Laplacian (second derivative)
    laplacian = cv.Laplacian(smoothed_image, cv.CV_64F)
    
    # Store the response squared (LoG response), as we are interested in extrema
    scale_space[:, :, i] = (sigma ** 2) * laplacian

# Find maxima in the scale space
blobs = np.zeros_like(gray_image)
for i in range(1, num_scales - 1):
    maxima = (scale_space[:, :, i] > scale_space[:, :, i - 1]) & (scale_space[:, :, i] > scale_space[:, :, i + 1])
    maxima = maxima & (scale_space[:, :, i] > threshold)
    blobs = np.logical_or(blobs, maxima)

# Detect blobs by non-maximum suppression
blobs = blobs.astype(np.uint8)

# Extract coordinates and radii of detected blobs (circles)
contours, _ = cv.findContours(blobs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
largest_circles = []
for contour in contours:
    (x, y), radius = cv.minEnclosingCircle(contour)
    largest_circles.append((int(x), int(y), int(radius)))

# Sort the circles by radius and pick the largest ones
largest_circles = sorted(largest_circles, key=lambda x: x[2], reverse=True)

# Draw the largest circles
for circle in largest_circles[:8]:  # Top 5 largest circles
    cv.circle(im, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

# Display the image with circles
plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
plt.title('Detected Circles')
plt.axis('off')
plt.show()

# Report the largest circles and σ values used
print("Largest Circles (x, y, radius):")
for circle in largest_circles[:8]:
    print(f"Circle at (x={circle[0]}, y={circle[1]}) with radius={circle[2]} pixels")

print(f"Range of sigma values used: {min_sigma} to {max_sigma}")
