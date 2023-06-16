import cv2
import numpy as np
import os
import glob

# Paths
input_path = r'data\data_resized\test\DRUSEN'
output_path = r'data\average_NORMAL.jpeg'

# Find all jpeg images in the folder
images_path = glob.glob(os.path.join(input_path, '*.jpeg'))

# Load all images
images = [cv2.imread(image) for image in images_path]

# Check if there's at least one image
if images:
    # Check if all images are of the same size
    image_shape = images[0].shape
    for img in images[1:]:
        if img.shape != image_shape:
            raise ValueError("All images must be of the same size")

    # Calculate the average of the images
    avg_image = np.mean(images, axis=0)

    # Save the resulting image
    cv2.imwrite(output_path, avg_image)
else:
    print("No images found in the directory.")
