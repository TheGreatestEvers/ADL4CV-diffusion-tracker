import torch
import time
import numpy as np

class ResidualFeatureBlock(torch.nn.Module):

    def __init__(self, n_input_channels: int = 3, intermediate_channels: list = [], n_output_channels: int = 128):
        """
        Parameters:
            n_input_channels (int): Number of input channels. Default is 3.
            intermediate_channels (list): List of intermediate channels for the convolutional layers.
            n_output_channels (int): Number of output channels. Default is 130.
        """
        super().__init__()

        self.channels = [n_input_channels, *intermediate_channels]
        self.n_output_channels = n_output_channels
        self.layers = torch.nn.ModuleList()
        
        for i in range(len(self.channels) - 1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=5, stride=1, padding='same'),
                torch.nn.BatchNorm2d(self.channels[i+1]),
                torch.nn.ReLU()
            )

            self.layers.append(layer)

        layer = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels[-1], self.n_output_channels, kernel_size=5, stride=1, padding='same'),
            torch.nn.BatchNorm2d(self.n_output_channels)
        )
        self.layers.append(layer)

    def forward(self, video: torch.Tensor):
        """
        Parameters:
            video (torch.Tensor): Input video tensor of shape (F, 3, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (F, n_output_channels, H, W) after passing through the block.
        """

        features = video

        for layer in self.layers:
            features = layer(features)

        return features