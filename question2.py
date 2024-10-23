import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to calculate line distance
def line_distance(params, points):
    a, b, d = params
    distances = np.abs(a * points[:, 0] + b * points[:, 1] - d) / np.sqrt(a**2 + b**2)
    return distances

# RANSAC for line fitting
def ransac_line(points, num_iterations=1000, distance_threshold=0.1, min_inliers=50):
    best_line = None
    best_inliers = []
    
    for i in range(num_iterations):
        # Randomly select two points
        sample_indices = np.random.choice(len(points), 2, replace=False)
        sample_points = points[sample_indices]
        
        # Fit a line to the two points
        a, b = sample_points[1, 1] - sample_points[0, 1], sample_points[0, 0] - sample_points[1, 0]
        d = a * sample_points[0, 0] + b * sample_points[0, 1]
        line_params = [a, b, d]
        
        # Normalize the parameters to ensure unit vector constraint
        norm = np.sqrt(a**2 + b**2)
        line_params = [a / norm, b / norm, d / norm]
        
        # Compute the distance of all points to the line
        distances = line_distance(line_params, points)
        inliers = points[distances < distance_threshold]
        
        # Update the best line if it has more inliers
        if len(inliers) > len(best_inliers):
            best_line = line_params
            best_inliers = inliers
            
        if len(best_inliers) > min_inliers:
            break
    
    return best_line, best_inliers

# Function to calculate circle distance
def circle_distance(params, points):
    x0, y0, r = params
    distances = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2) - r
    return np.abs(distances)

# RANSAC for circle fitting
def ransac_circle(points, num_iterations=1000, distance_threshold=0.1, min_inliers=50):
    best_circle = None
    best_inliers = []
    
    for i in range(num_iterations):
        # Randomly select three points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]
        
        # Solve for the circle that passes through the three points
        def circle_equation(params):
            distances = circle_distance(params, sample_points)
            return np.sum(distances**2)  # Sum of squared distances

        
        initial_guess = [0, 0, 10]
        result = minimize(circle_equation, initial_guess)
        circle_params = result.x
        
        # Compute the distance of all points to the circle
        distances = circle_distance(circle_params, points)
        inliers = points[distances < distance_threshold]
        
        # Update the best circle if it has more inliers
        if len(inliers) > len(best_inliers):
            best_circle = circle_params
            best_inliers = inliers
            
        if len(best_inliers) > min_inliers:
            break
    
    return best_circle, best_inliers

# Plotting function
def plot_results(points, line_params, circle_params, line_inliers, circle_inliers, ground_truth_circle):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], label='Noisy Points', color='gray', alpha=0.5)
    
    # Plot line inliers and best line
    ax.scatter(line_inliers[:, 0], line_inliers[:, 1], label='Line Inliers', color='blue')
    x_vals = np.array(ax.get_xlim())
    y_vals = (line_params[2] - line_params[0] * x_vals) / line_params[1]
    ax.plot(x_vals, y_vals, 'b-', label='Best Fit Line')
    
    # Plot circle inliers and best circle
    ax.scatter(circle_inliers[:, 0], circle_inliers[:, 1], label='Circle Inliers', color='red')
    circle = plt.Circle((circle_params[0], circle_params[1]), circle_params[2], color='red', fill=False)
    ax.add_artist(circle)
    
    # Plot ground truth circle
    ground_truth_circle_artist = plt.Circle((ground_truth_circle[0], ground_truth_circle[1]), ground_truth_circle[2], color='green', fill=False, linestyle='--', label='Ground Truth Circle')
    ax.add_artist(ground_truth_circle_artist)
    
    plt.legend()
    plt.show()

# Main code
N = 100
half_n = N // 2
r = 10
x0_gt, y0_gt = 2, 3  # Circle center
s = r / 16
t = np.random.uniform(0, 2 * np.pi, half_n)
n = s * np.random.randn(half_n)
x, y = x0_gt + (r + n) * np.cos(t), y0_gt + (r + n) * np.sin(t)
X_circ = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

s = 1.0
m, b = -1, 2
x = np.linspace(-20, 20, half_n)
y = m * x + b + s * np.random.randn(half_n)
X_line = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

X = np.vstack((X_circ, X_line))

# Fit line using RANSAC
line_params, line_inliers = ransac_line(X)

# Remove line inliers and fit circle using RANSAC
remaining_points = np.array([pt for pt in X if pt not in line_inliers])
circle_params, circle_inliers = ransac_circle(remaining_points)

# Ground truth circle
ground_truth_circle = [x0_gt, y0_gt, r]
# Plot the results
plot_results(X, line_params, circle_params, line_inliers, circle_inliers, ground_truth_circle)
