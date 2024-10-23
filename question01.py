import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def laplace_of_gaussian_filter(s):
    hw_size = round(3 * s)  # Half-width of kernel
    X, Y = np.meshgrid(np.arange(-hw_size, hw_size + 1, 1), np.arange(-hw_size, hw_size + 1, 1))

    log_kernel = ((X**2 + Y**2) / (2 * s**2) - 1) * np.exp(-(X**2 + Y**2) / (2 * s**2)) / (np.pi * s**4)
    return log_kernel

def find_local_maximums(log_image, sigma_value):
    maxima_coords = []
    (img_height, img_width) = log_image.shape
    kernel_size = 1
    for row in range(kernel_size, img_height - kernel_size):
        for col in range(kernel_size, img_width - kernel_size):
            sub_image = log_image[row - kernel_size:row + kernel_size + 1, col - kernel_size:col + kernel_size + 1]
            max_value = np.max(sub_image)  # Finding the local maximum
            if max_value >= 0.09:  # Threshold
                max_row, max_col = np.unravel_index(sub_image.argmax(), sub_image.shape)
                maxima_coords.append((row + max_row - kernel_size, col + max_col - kernel_size))  # Append coordinates
    return set(maxima_coords)

# Load and normalize the grayscale image
image = cv.imread('images/the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) / 255

# Create a 3x3 grid for visualizing results
fig, axis_grid = plt.subplots(3, 3, figsize=(15, 15))

# Iterate over different sigma values
for r_value, ax_subplot in enumerate(axis_grid.flatten(), start=1):
    sigma_value = r_value / 1.414
    laplace_gaussian = sigma_value**2 * laplace_of_gaussian_filter(sigma_value)
    log_image = np.square(cv.filter2D(image, -1, laplace_gaussian))

    # Detect local maxima
    maxima_coords = find_local_maximums(log_image, sigma_value)

    ax_subplot.imshow(log_image, cmap='gray')
    ax_subplot.set_title(f'r = {r_value}')

    # Mark detected blobs
    for x_val, y_val in maxima_coords:
        circle = plt.Circle((y_val, x_val), sigma_value * 1.414, color='red', linewidth=1, fill=False)
        ax_subplot.add_patch(circle)
    ax_subplot.plot()

plt.axis('off')
plt.show()

# Reload and normalize image for further processing
original_image = cv.imread('images/the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY) / 255

# Create a figure with two subplots
fig, ax_list = plt.subplots(1, 2, figsize=(10, 10))

# Display the original image
ax_list[0].imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))
ax_list[0].set_title("Original Image")
ax_list[0].axis('off')

# Display the grayscale image with detected blobs
second_ax = ax_list[1]
second_ax.imshow(gray_image, cmap='gray')
second_ax.grid(False)

# Create patches and labels for the legend
color_options = list(mcolors.TABLEAU_COLORS)
patch_items = []
label_items = []

# Iterate over a range of sigma values
for r_value in range(1, 11):
    sigma_value = r_value / 1.414
    laplace_gaussian = sigma_value**2 * laplace_of_gaussian_filter(sigma_value)
    log_image = np.square(cv.filter2D(gray_image, -1, laplace_gaussian))

    # Detect local maxima
    maxima_coords = find_local_maximums(log_image, sigma_value)

    # Mark detected blobs with circles
    for x_val, y_val in maxima_coords:
        circle = plt.Circle((y_val, x_val), sigma_value * 1.414, color=color_options[r_value - 1], linewidth=1, fill=False)
        second_ax.add_patch(circle)
    patch_items.append(circle)
    label_items.append(f'r = {r_value}')
    second_ax.plot()

second_ax.set_xlim(0, original_image.shape[1])

plt.axis('off')
second_ax.legend(patch_items, label_items, loc='best', fontsize=8)
second_ax.set_title("Detected blobs at different sigma values")
plt.show()
