from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    Applies two consecutive convolution operations with batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        conv (nn.Sequential): Sequence of convolution, batch normalization, and ReLU layers.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the double convolution operation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)


class UNET(nn.Module):
    """
    U-Net model for image segmentation.

    Args:
        in_channels (int): Number of input channels. Default is 3 (RGB images).
        out_channels (int): Number of output channels. Default is 1 (single channel output).
        features (List[int]): List of feature channels. Default is [64, 128, 256, 512].

    Attributes:
        ups (nn.ModuleList): List of upsampling and double convolution layers.
        downs (nn.ModuleList): List of downsampling and double convolution layers.
        pool (nn.MaxPool2d): Max pooling layer for downsampling.
        input_size (Tuple[int, int]): Target input size for the U-Net.
        bottleneck (DoubleConv): Double convolution layer at the bottleneck of the U-Net.
        final_conv (nn.Conv2d): Final convolution layer to produce the output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
    ) -> None:
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.input_size: Tuple[int, int] = (256, 256)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the U-Net model to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Resize the input tensor to 256x256
        x = TF.resize(x, size=self.input_size)

        skip_connections: List[torch.Tensor] = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Resize the skip connection to match the current feature map size
            if x.shape != skip_connection.shape:
                skip_connection = TF.resize(skip_connection, size=x.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
