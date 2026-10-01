# models/hsi_rgb_adapter.py
import torch
import torch.nn as nn

class HSIRGBAdapter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3)
        )

    def forward(self, x):
        # x: (B, Bands, 13, 13)
        return self.proj(x)
