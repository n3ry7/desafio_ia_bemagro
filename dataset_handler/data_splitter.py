import os
import random
import shutil
from typing import List, Tuple


class DataSplitter:
    """
    A class that splits a dataset of images and masks into training and validation sets.

    Attributes:
        image_dir (str): The path to the directory containing the input images.
        masks_dir (str): The path to the directory containing the corresponding binary masks.
        train_image_dir (str): The path to the directory where the training images will be stored.
        train_mask_dir (str): The path to the directory where the training masks will be stored.
        val_image_dir (str): The path to the directory where the validation images will be stored.
        val_mask_dir (str): The path to the directory where the validation masks will be stored.
    """

    def __init__(self, image_dir: str, masks_dir: str):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.train_image_dir = "dataset/train_images"
        self.train_mask_dir = "dataset/train_masks"
        self.val_image_dir = "dataset/val_images"
        self.val_mask_dir = "dataset/val_masks"

    def apply(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Splits the dataset into training and validation sets, and copies the
        images and masks to the corresponding directories.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: A tuple containing
            the lists of training image files, training mask files, validation image files,
            and validation mask files.
        """
        # Get a list of all image files in the source directory
        image_files = os.listdir(self.image_dir)

        # Select 10% of the images at random
        num_images = len(image_files)
        num_val_images = int(num_images * 0.1)
        val_indices = random.sample(range(num_images), num_val_images)

        # Create the destination directories
        os.makedirs(self.train_image_dir, exist_ok=True)
        os.makedirs(self.train_mask_dir, exist_ok=True)
        os.makedirs(self.val_image_dir, exist_ok=True)
        os.makedirs(self.val_mask_dir, exist_ok=True)

        train_image_files = []
        train_mask_files = []
        val_image_files = []
        val_mask_files = []

        # Copy the images and masks to the destination directories
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(
                self.masks_dir, os.path.splitext(image_file)[0] + "_plant_mask.png"
            )
            if i in val_indices:
                shutil.copy(image_path, os.path.join(self.val_image_dir, image_file))
                shutil.copy(
                    mask_path,
                    os.path.join(
                        self.val_mask_dir, os.path.splitext(image_file)[0] + "_plant_mask.png"
                    ),
                )
                val_image_files.append(image_file)
                val_mask_files.append(os.path.splitext(image_file)[0] + "_plant_mask.png")
            else:
                shutil.copy(image_path, os.path.join(self.train_image_dir, image_file))
                shutil.copy(
                    mask_path,
                    os.path.join(
                        self.train_mask_dir, os.path.splitext(image_file)[0] + "_plant_mask.png"
                    ),
                )
                train_image_files.append(image_file)
                train_mask_files.append(os.path.splitext(image_file)[0] + "_plant_mask.png")

        print("Data split complete.")
        return train_image_files, train_mask_files, val_image_files, val_mask_files
