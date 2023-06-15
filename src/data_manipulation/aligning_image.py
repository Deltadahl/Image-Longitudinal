import cv2
from skimage import io, filters, img_as_ubyte

def preprocess_image(image_path, output_path):
    # Load image
    image = io.imread(image_path, as_gray=True)

    # Apply Otsu's thresholding
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold

    # Convert the binary image back to the range [0-255]
    binary_image = img_as_ubyte(binary_image)

    # Write the binary image to the output path
    cv2.imwrite(output_path, binary_image)

# Test the function
input_path = r'data\CellData\OCT_mb3d\train\CNV\CNV-172472-150.jpeg'
output_path = r'data\TESTING_alignment.jpeg'
preprocess_image(input_path, output_path)
