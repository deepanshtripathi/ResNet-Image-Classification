import os
import cv2
import numpy as np
from tqdm import tqdm  # Adds a progress bar for tracking progress
from pathlib import Path

# Define paths for raw and processed data
ROOT_DIR = Path(__file__).resolve().parent.parent  # Get the absolute path of the project root
RAW_DATA_PATH = ROOT_DIR / "data/raw"  # Directory where the raw dataset is stored
PROCESSED_DATA_PATH = ROOT_DIR / "data/processed"  # Directory where the processed images will be saved

# Image processing parameters
IMG_SIZE = 64  # Standardizing image size to 64x64 pixels for consistency in training


def preprocess_and_save_images():
    """
    This function reads images from the raw dataset, resizes them to 64x64 pixels,
    converts them to grayscale, and saves them into the processed dataset directory.
    """

    # Ensure the processed data directory exists before saving processed images
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Iterate over dataset splits: training, testing, and validation
    for split in ["train", "test", "val"]:
        raw_split_path = RAW_DATA_PATH / split  # Path to raw data (e.g., data/raw/train)
        processed_split_path = PROCESSED_DATA_PATH / split  # Path to processed data (e.g., data/processed/train)
        processed_split_path.mkdir(exist_ok=True)  # Create directory if it does not exist

        # Iterate over class labels (cat, dog)
        for category in ["cat", "dog"]:
            raw_category_path = raw_split_path / category  # Path to raw class images (e.g., data/raw/train/cat)
            processed_category_path = processed_split_path / category  # Path to processed images (e.g., data/processed/train/cat)
            processed_category_path.mkdir(exist_ok=True)  # Ensure the processed class directory exists
            tqdm.write(f"\nProcessing {split}/{category} images - ")

            # Process each image in the category folder
            for img_name in tqdm(os.listdir(raw_category_path)):  # tqdm adds a progress bar
                img_path = raw_category_path / img_name  # Full path to the raw image
                processed_img_path = processed_category_path / img_name  # Destination path for processed image

                # Read image in grayscale mode to reduce complexity
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                # Handle corrupted or unreadable images
                if img is None:
                    print(f"Skipping corrupted image - {img_path}")
                    continue

                # Resize the image to a fixed size (64x64) for consistency across the dataset
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Save the processed image in the respective directory
                cv2.imwrite(str(processed_img_path), img_resized)

    # Confirm completion of preprocessing
    print("\nPreprocessing completed. Processed images saved in 'data/processed/'")


if __name__ == "__main__":
    preprocess_and_save_images()
