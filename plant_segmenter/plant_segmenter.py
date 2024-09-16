import torch
import torchvision.transforms as transforms

from plant_segmenter.model.u_net import UNET


class PlantSegmenter:
    """
    A class for performing plant segmentation using a pre-trained U-Net model.

    Attributes:
        device (torch.device): The device on which the model will be run (CPU or GPU).
        model (UNET): The U-Net model used for plant segmentation.
        preprocess (torchvision.transforms.Compose): The preprocessing transformations
        applied to the input image.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the PlantSegmenter class.

        Args:
            model_path (str): The path to the pre-trained U-Net model checkpoint.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNET(in_channels=3, out_channels=1).to(self.device)
        self.load_checkpoint(model_path)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )

    def load_checkpoint(self, model_path: str) -> None:
        """
        Loads the pre-trained U-Net model checkpoint.

        Args:
            model_path (str): The path to the pre-trained U-Net model checkpoint.
        """
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def segment(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Performs plant segmentation on the input RGB image.

        Args:
            rgb_image (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The segmented plant image, where 1 represents a plant pixel
            and 0 represents a non-plant pixel.
        """
        with torch.no_grad():
            image = self.preprocess(rgb_image).unsqueeze(0).to(self.device)
            prediction = self.model(image)
            segmented_image = (prediction > 0.5).float().squeeze(0).cpu()
        return segmented_image
