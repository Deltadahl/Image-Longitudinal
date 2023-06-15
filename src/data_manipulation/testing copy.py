import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from skimage.util import random_noise

def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    # Convert input image to float
    img = img_as_float(img)

    # PDE (partial differential equation) initial condition
    u = img

    # Center pixel distances
    dx = dy = dz = 1

    for t in range(niter):
        # 3D gradients
        ux, uy = np.gradient(u)

        # Diffusion function
        if option == 1:
            g = np.ones(u.shape)
        elif option == 2:
            g = 1 / (1 + (np.sqrt(ux**2 + uy**2) / kappa)**2)

        # Update matrices
        D = g

        # Finite difference scheme
        u += gamma * (D * np.gradient(ux)[0] + D * np.gradient(uy)[1])

    return u

def flatten_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Denoise the image using BM3D
    # The denoising function is not implemented here and should be added
    # denoised_img = BM3D(img)

    # Apply anisotropic diffusion
    img_filtered = anisotropic_diffusion(img)

    # Apply adaptive thresholding
    threshold_value = 44
    _, binarized_img = cv2.threshold(img_filtered, threshold_value, 255, cv2.THRESH_BINARY)

    # Get the uppermost and lowermost white pixel
    white_pixel_coords = np.where(binarized_img == 255)
    uppermost_white_pixel = np.min(white_pixel_coords[0])
    lowermost_white_pixel = np.max(white_pixel_coords[0])

    # Crop the image horizontally
    cropped_img = img[uppermost_white_pixel:lowermost_white_pixel+1, :]

    # Here, you can save the cropped_img or return it
    cv2.imwrite('TESTING_cropped_image.png', cropped_img)

img_path = r"data\CellData\OCT_mb3d\train\NORMAL\NORMAL-3730077-14.jpeg"
flatten_image(img_path)
