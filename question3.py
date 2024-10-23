import cv2 as opencv
import numpy as np
import matplotlib.pyplot as plt

# Load images
architecture_image = opencv.imread('Images/03.jpg')  # Replace with your architectural image
national_flag_image = opencv.imread('Images/images.jpg')  # Replace with your flag image

# Resize the flag image (optional, adjust as needed)
national_flag_image = opencv.resize(national_flag_image, (200, 100))

# Store the points selected by the user
user_selected_points = []

# Mouse callback function to select points on the image
def capture_point(event, x, y, flags, param):
    if event == opencv.EVENT_LBUTTONDOWN and len(user_selected_points) < 4:
        user_selected_points.append((x, y))
        print(f"Selected point: {x, y}")

# Display the building image and allow the user to select points
opencv.imshow('Select 4 points on the building image', architecture_image)
opencv.setMouseCallback('Select 4 points on the building image', capture_point)

# Wait until 4 points are selected
while len(user_selected_points) < 4:
    opencv.waitKey(1)

opencv.destroyAllWindows()

# Coordinates of the four corners of the flag (source points in the flag image)
flag_corners = np.float32([
    [0, 0], 
    [national_flag_image.shape[1], 0], 
    [national_flag_image.shape[1], national_flag_image.shape[0]], 
    [0, national_flag_image.shape[0]]
])

# Coordinates of the points clicked on the building image (destination points)
building_corners = np.float32(user_selected_points)

# Compute homography matrix
homography_matrix, status = opencv.findHomography(flag_corners, building_corners)

# Warp the flag image to align with the selected building points
warped_flag_image = opencv.warpPerspective(national_flag_image, homography_matrix, (architecture_image.shape[1], architecture_image.shape[0]))

# Create a mask from the warped flag to blend it with the building image
masking_image = np.zeros_like(architecture_image, dtype=np.uint8)
opencv.fillConvexPoly(masking_image, np.int32(building_corners), (255, 255, 255))

# Inverse the mask to remove the area where the flag will be placed
architecture_masked_image = opencv.bitwise_and(architecture_image, opencv.bitwise_not(masking_image))

# Blend the warped flag onto the building image
final_result = opencv.add(architecture_masked_image, warped_flag_image)

# Display the result
plt.imshow(opencv.cvtColor(final_result, opencv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Result with Flag Overlay')
plt.show()

# Save the result if needed
opencv.imwrite('result_with_flag.jpg', final_result)
