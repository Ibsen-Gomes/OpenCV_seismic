import torch
import torch.nn as nn
import torch.nn.functional as F

class SeismicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3):
        super(SeismicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn

class CNNSeismicClassifierV3(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNSeismicClassifierV3, self).__init__()

        self.encoder = nn.Sequential(
            SeismicBlock(1, 32),
            nn.MaxPool2d(2),

            SeismicBlock(32, 64),
            nn.MaxPool2d(2),

            SeismicBlock(64, 128, dilation=2),
            nn.MaxPool2d(2),

            SeismicBlock(128, 256, dilation=2),
            SpatialAttention(),  # foca nas áreas importantes
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
