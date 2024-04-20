from datasets import Dataset, DatasetDict
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from tqdm import tqdm
import time
import gc
import numpy as np
import json
import re

initial_time = time.time()

# Define image and numeric folder paths
image_folder_path = "../data/TRAIN_DATA_2"
numeric_folder_path = "../TRAIN_DATA_2_json"

def get_image_files(root):
    for path, subdirs, files in os.walk(root):
        if "_test" in path or "_json" in path:
            print(f"Skipping {path}")
            continue

        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                relative_path = os.path.relpath(path, root)

                json_file_path = os.path.join(numeric_folder_path, relative_path, name.split(".")[0] + ".json")
                # json_file_path = "../data/zero.json"
                if os.path.exists(json_file_path):
                    yield os.path.join(path, name), json_file_path
                else:
                    print(f"NOT Found {json_file_path}")

def load_image(file_path, json_path):
    img = Image.open(file_path)
    width, height = img.size

    if width > 2000 or height > 2000:
        return None, None

    img = img.convert("RGB").resize((512, 512))  # note the size!

    with open(json_path, 'r') as f:
        numeric = json.load(f)
        numeric = np.array(numeric, dtype=np.float32)

    return img, numeric

def process_images(image_files):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(load_image, file_path, json_path): (file_path, json_path)
            for file_path, json_path in image_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
            result = future.result()
            if result[0] is not None:
                results.append(result)
        return results


# Function to split the list into chunks
def split_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


base_path = f"data_arrow_OCT"  # NOTE
if not os.path.exists(base_path):
    os.makedirs(base_path)

chunk_size = 30000
# Load all image file names and their folder names
image_files = list(get_image_files(image_folder_path))
# Split the image files into chunks of chunk_size
chunks = list(split_into_chunks(image_files, chunk_size))

for index, chunk in enumerate(chunks):
    dataset_path = os.path.join(base_path, f"dataset_chunk_{index + 1}")

    # Check if the dataset for this chunk already exists
    if os.path.exists(dataset_path):
        print(f"Skipping already processed chunk {index + 1}")
        continue

    processed_images_texts = process_images(chunk)

    # Convert lists to a dictionary
    data = {
        "image": [img for img, text in processed_images_texts],
        "text": [text for img, text in processed_images_texts],
    }

    # Convert dictionary to Hugging Face Dataset
    dataset = Dataset.from_dict(data)

    # Create a DatasetDict with training set
    dataset_dict = DatasetDict({"train": dataset})

    # Save the dataset to disk
    dataset_dict.save_to_disk(dataset_path)

    print(f"Time taken: {time.time() - initial_time}")
    print(f"Number of images processed: {(index + 1) * chunk_size}")

    del processed_images_texts, data, dataset, dataset_dict
    gc.collect()  # Explicitly call garbage collector

n_chunks = len(chunks)

# Load and concatenate datasets
all_datasets = []
for i in range(1, n_chunks + 1):
    dataset = load_from_disk(f"{base_path}/dataset_chunk_{i}")["train"]
    all_datasets.append(dataset)

if len(all_datasets) == 1:
    merged_train_dataset = all_datasets[0]
else:
    merged_train_dataset = concatenate_datasets(all_datasets)

# Create a new DatasetDict with the merged 'train' dataset
merged_dataset_dict = DatasetDict({"train": merged_train_dataset})

# Save the merged DatasetDict to disk
merged_dataset_dict.save_to_disk(f"{base_path}/dataset_with_text_MERGED")
