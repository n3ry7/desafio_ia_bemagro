import os

import albumentations
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from decouple import config

from plant_segmenter.model.u_net import UNET
from plant_segmenter.utils.train_utils import (
    check_accuracy,
    get_loaders,
    load_checkpoint,
    save_checkpoint,
)


class UNetTrainer:
    """A class for training a U-Net model on image segmentation tasks.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        device (str): The device to use for training and inference ('cuda' or 'cpu').
        batch_size (int): The batch size for training and validation.
        num_epochs (int): The number of epochs to train the model.
        num_workers (int): The number of workers to use for data loading.
        image_height (int): The height of the input images.
        image_width (int): The width of the input images.
        pin_memory (bool): Whether to use pin memory for data loading.
        load_model (bool): Whether to load a pre-trained model.
        model_path (str): Path to save trained weights.
        train_img_dir (str): The directory containing the training images.
        train_mask_dir (str): The directory containing the training masks.
        val_img_dir (str): The directory containing the validation images.
        val_mask_dir (str): The directory containing the validation masks.
    """

    def __init__(
        self,
        learning_rate: float = config("LEARNING_RATE", cast=float, default=1e-4),
        device: str = config("DEVICE", default="cuda" if torch.cuda.is_available() else "cpu"),
        batch_size: int = config("BATCH_SIZE", cast=int, default=32),
        num_epochs: int = config("NUM_EPOCHS", cast=int, default=6),
        num_workers: int = config("NUM_WORKERS", cast=int, default=4),
        image_height: int = config("IMAGE_HEIGHT", cast=int, default=256),
        image_width: int = config("IMAGE_WIDTH", cast=int, default=256),
        pin_memory: bool = config("PIN_MEMORY", cast=bool, default=True),
        load_model: bool = config("LOAD_MODEL", cast=bool, default=False),
        model_path: str = config("MODEL_PATH", default="plant_segmenter/weights/u_net_weights.pth"),
        train_img_dir: str = config("TRAIN_IMG_DIR", default="dataset/train_images"),
        train_mask_dir: str = config("TRAIN_MASK_DIR", default="dataset/train_masks"),
        val_img_dir: str = config("VAL_IMG_DIR", default="dataset/val_images"),
        val_mask_dir: str = config("VAL_MASK_DIR", default="dataset/val_masks"),
    ):
        self.learning_rate = learning_rate
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.image_height = image_height
        self.image_width = image_width
        self.pin_memory = pin_memory
        self.load_model = load_model
        self.model_path = model_path
        self.train_img_dir = train_img_dir
        self.train_mask_dir = train_mask_dir
        self.val_img_dir = val_img_dir
        self.val_mask_dir = val_mask_dir

    def train(self) -> None:
        """Train the U-Net model on the provided data."""
        train_transform = albumentations.Compose(
            [
                albumentations.Resize(height=self.image_height, width=self.image_width),
                albumentations.Rotate(limit=35, p=1.0),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.1),
                albumentations.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

        val_transforms = albumentations.Compose(
            [
                albumentations.Resize(height=self.image_height, width=self.image_width),
                albumentations.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

        model = UNET(in_channels=3, out_channels=1).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        train_loader, val_loader = get_loaders(
            self.train_img_dir,
            self.train_mask_dir,
            self.val_img_dir,
            self.val_mask_dir,
            self.batch_size,
            train_transform,
            val_transforms,
            self.num_workers,
            self.pin_memory,
        )

        if self.load_model:
            load_checkpoint(torch.load(self.model_path), model)

        check_accuracy(val_loader, model, device=self.device)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.num_epochs):
            self.train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # Save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, self.model_path)

            # Check accuracy
            check_accuracy(val_loader, model, device=self.device)

    def train_fn(
        self,
        loader: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scaler: torch.cuda.amp.GradScaler,
    ) -> None:
        """Train the model for one epoch.

        Args:
            loader (torch.utils.data.DataLoader): The data loader for the training data.
            model (nn.Module): The U-Net model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            loss_fn (nn.Module): The loss function used for training.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler used for mixed precision
            training.
        """
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.device)
            targets = targets.float().unsqueeze(1).to(device=self.device)

            # Forward pass
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update tqdm loop
            loop.set_postfix(loss=loss.item())
