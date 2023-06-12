import os
import numpy as np
from skimage import io, img_as_float, img_as_ubyte, color
import time

import bm3d

def bm3d_denoise(input_image_path, output_image_path):
    # Read the noisy image
    noisy_image = img_as_float(io.imread(input_image_path))

    # Convert the image to grayscale if it's not
    if len(noisy_image.shape) > 2:
        noisy_image = color.rgb2gray(noisy_image)

    # Perform BM3D denoising
    denoised_image = bm3d.bm3d(noisy_image, sigma_psd=0.3, stage_arg=bm3d.BM3DStages.ALL_STAGES)

    # Normalize the denoised image
    denoised_image = (denoised_image - np.min(denoised_image)) / (np.max(denoised_image) - np.min(denoised_image))

    # Convert the normalized image data to 8-bit format
    denoised_image = img_as_ubyte(denoised_image)

    # make path if it doesn't exist, then save image
    if not os.path.exists(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))

    # saving image
    io.imsave(output_image_path, denoised_image)

def process_directory(input_directory, output_directory):
    # Loop over all files in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a JPEG image
        if filename.lower().endswith(".jpeg"):
            # Construct full input and output paths
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            # Perform BM3D denoising and save the output
            bm3d_denoise(input_path, output_path)

if __name__ == "__main__":
    time_start = time.time()
    input_directory = "data\CellData\OCT_white_to_black\DEVELOP"
    output_directory = "data\CellData\PYTHON_TEST"
    process_directory(input_directory, output_directory)
    time_end = time.time()
    print("Time elapsed: ", time_end - time_start)
