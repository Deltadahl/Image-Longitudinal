import cv2
import numpy as np
from scipy import optimize
from skimage import img_as_ubyte, morphology, filters
from skimage.color import rgb2gray
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.signal import medfilt
from scipy.stats import pearsonr

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def flatten_image(input_path, output_path):
    # Load the image
    image = cv2.imread(input_path)

    # Convert the image to grayscale
    gray = rgb2gray(image)
    gray = img_as_ubyte(gray)

    # Thresholding
    threshold = filters.threshold_otsu(gray)
    binary = gray > threshold
    binary = binary.astype(float)

    # Median filtering
    binary = medfilt(binary, kernel_size=3)

    # Morphological closing and opening
    binary = morphology.closing(binary)
    binary = morphology.opening(binary)

    # Calculate middle and bottom points
    y, x = np.where(binary)
    x_values, index = np.unique(x, return_inverse=True)
    middle_points = np.bincount(index, weights=y) / np.bincount(index)
    # Calculate bottom points
    indices = np.r_[np.unique(index, return_index=True)[1], len(index)]
    indices = indices[indices < len(y)]  # Ensure indices do not exceed the length of y
    bottom_points = np.maximum.reduceat(y, indices)
    # bottom_points = np.maximum.reduceat(y, np.r_[np.unique(index, return_index=True)[1], len(index)])

    # Generate x-values
    x_values = np.arange(binary.shape[1])

    # Choose the set of points and fitting function
    params, params_covariance = optimize.curve_fit(quadratic, x_values, middle_points, p0=[1, 1, 1])

    if params[0] < 0:
        data_points = bottom_points
    else:
        data_points = middle_points

    params_linear, params_covariance_linear = optimize.curve_fit(linear, x_values, data_points, p0=[1, 1])
    params_quadratic, params_covariance_quadratic = optimize.curve_fit(quadratic, x_values, data_points, p0=[1, 1, 1])

    fitted_linear = linear(x_values, params_linear[0], params_linear[1])
    fitted_quadratic = quadratic(x_values, params_quadratic[0], params_quadratic[1], params_quadratic[2])

    if pearsonr(data_points, fitted_linear)[0] > pearsonr(data_points, fitted_quadratic)[0]:
        fitted_values = fitted_linear
    else:
        fitted_values = fitted_quadratic

    # Normalize the image
    for i in range(image.shape[1]):
        image[:, i] = np.roll(image[:, i], -int(fitted_values[i]), axis=0)

    # Save the flattened image
    cv2.imwrite(output_path, image)

# Use the function
input_path = r"data\CellData\OCT_mb3d\train\NORMAL\NORMAL-586534-7.jpeg"
output_path = r"data\TESTING.jpeg"
flatten_image(input_path, output_path)
