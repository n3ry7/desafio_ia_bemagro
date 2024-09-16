import os
from pathlib import Path
from typing import Tuple

import torch
import torchvision
from torch.utils.data import DataLoader

from plant_segmenter.dataset import PlantDataset


def save_checkpoint(state: dict, filename: str = "u_net_weights.pth") -> None:
    """
    Saves the current state of the model to a checkpoint file.

    Args:
        state (dict): A dictionary containing the current state of the model.
        filename (str, optional): The name of the checkpoint file.
        Defaults to "u_net_weights.pth".
    """
    file_path = os.path.join(os.getcwd(), filename)
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("=> Saving checkpoint")
    torch.save(state, file_path)


def load_checkpoint(checkpoint: dict, model: torch.nn.Module) -> None:
    """
    Loads the state of the model from a checkpoint file.

    Args:
        checkpoint (dict): A dictionary containing the saved state of the model.
        model (torch.nn.Module): The model to be loaded with the saved state.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir: str,
    train_maskdir: str,
    val_dir: str,
    val_maskdir: str,
    batch_size: int,
    train_transform: torchvision.transforms.Compose,
    val_transform: torchvision.transforms.Compose,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates the training and validation data loaders.

    Args:
        train_dir (str): The directory containing the training images.
        train_maskdir (str): The directory containing the training masks.
        val_dir (str): The directory containing the validation images.
        val_maskdir (str): The directory containing the validation masks.
        batch_size (int): The batch size for the data loaders.
        train_transform (torchvision.transforms.Compose): The transform to be applied to
        the training data.
        val_transform (torchvision.transforms.Compose): The transform to be applied to the
        validation data.
        num_workers (int, optional): The number of worker processes to use for data loading.
        Defaults to 4.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA
        pinned memory before returning them. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader]: The training and validation data loaders.
    """
    train_ds = PlantDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = PlantDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(
    loader: DataLoader,
    model: torch.nn.Module,
    device: str = "cuda",
) -> None:
    """
    Checks the accuracy of the model on the given data loader.

    Args:
        loader (DataLoader): The data loader to use for checking accuracy.
        model (torch.nn.Module): The model to be evaluated.
        device (str, optional): The device to use for the model. Defaults to "cuda".
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader: DataLoader,
    model: torch.nn.Module,
    folder: str = "saved_images/",
    device: str = "cuda",
) -> None:
    """
    Saves the predictions of the model as images in the specified folder.

    Args:
        loader (DataLoader): The data loader to use for generating the predictions.
        model (torch.nn.Module): The model to be used for making the predictions.
        folder (str, optional): The folder to save the images in. Defaults to "saved_images/".
        device (str, optional): The device to use for the model. Defaults to "cuda".
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
