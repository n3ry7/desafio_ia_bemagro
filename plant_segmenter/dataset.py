import os
from typing import Optional, Tuple

import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PlantDataset(Dataset):
    """
    A PyTorch dataset class for the plant image and mask data.

    Args:
        image_dir (str): The directory containing the plant images.
        mask_dir (str): The directory containing the plant masks.
        transform (Optional[albumentations.Compose]): The data
        augmentation pipeline to be applied to
        the images and masks.
    """

    def __init__(
        self, image_dir: str, mask_dir: str, transform: Optional[albumentations.Compose] = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the
            image and mask data for the sample.
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".png", "_plant_mask.png")
        )
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
