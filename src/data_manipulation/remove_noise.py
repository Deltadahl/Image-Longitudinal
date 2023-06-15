import os
import time
import imageio
import numpy as np
from bm3d import bm3d, BM3DStages
from glob import glob

# Function to denoise the image
def denoise_image(img, noise_variance):
    img = img.astype(np.float64)/255  # normalize the data to 0 - 1
    # img_denoised = bm3d(img, noise_variance)
    img_denoised = bm3d(img, noise_variance, stage_arg=BM3DStages.ALL_STAGES)
    img_denoised = (img_denoised - np.min(img_denoised)) / (np.max(img_denoised) - np.min(img_denoised))  # Normalize
    img_denoised *= 255  # scale back to 0 - 255
    return img_denoised.astype(np.uint8)

# Function to process each file
def process_file(file, noise_variance, base_path, base_path_modified, counter):
    # Maintain the subfolder structure in the new path
    relative_path = os.path.relpath(file, base_path)
    save_path = os.path.join(base_path_modified, relative_path)

    # Check if the file already exists in the output directory. If it does, then continue with the next iteration.
    if os.path.isfile(save_path):
        print(f"File already exists -> Skipping: {file}")
        return counter

    img = imageio.imread(file)
    img_denoised = denoise_image(img, noise_variance)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imsave(save_path, img_denoised)

    return counter + 1


def main():
    base_path = "data/CellData/OCT_white_to_black"
    base_path_modified = "data/CellData/OCT_mb3d"
    noise_variance = 0.15
    subfolders = [
        "train/DRUSEN",
        "train/NORMAL",
        "train/CNV",
    ]

    print("Starting the main task")
    # Flatten the file structure
    all_files = [glob(os.path.join(base_path, subfolder, "*.jpeg")) for subfolder in subfolders]
    all_files = [file for sublist in all_files for file in sublist]  # flatten the list

    counter = 0
    start_time = time.time()
    for file in all_files:
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Counter {counter} Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        counter = process_file(file, noise_variance, base_path, base_path_modified, counter)

        print(f"Finished processing {file}")

if __name__ == "__main__":
    main()
