import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUpsampling1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            nn.ConvTranspose1d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=2, output_padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )
        self.residual = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels,
                               kernel_size=1, stride=stride,
                               output_padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class ResidualHeightUpsampling2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=5, stride=(2, 1),
                               padding=2, output_padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.residual = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=1, stride=(2, 1),
                               output_padding=(1, 0)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class ResidualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 7), padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, C_in, H, W)
        return self.conv(x) + self.residual(x)

class ResidualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (B, C_in, H, W)
        return self.conv(x) + self.residual(x)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_padding = nn.ConstantPad1d((1, 1), 0)  # L: 30 -> 32

        self.temporal_upsampling = nn.Sequential(
            ResidualConv1d(in_channels=768, out_channels=768),
            ResidualConv1d(768, 1024),
            ResidualUpsampling1d(in_channels=1024, out_channels=1024),  # L: 32 -> 64
            ResidualConv1d(1024, 1024),
            ResidualUpsampling1d(in_channels=1024, out_channels=1024),  # L: 64 -> 128
            ResidualConv1d(1024, 1024),
            ResidualUpsampling1d(in_channels=1024, out_channels=1024),  # L: 128 -> 256
            ResidualConv1d(1024, 1024),
            ResidualUpsampling1d(in_channels=1024, out_channels=1024),  # L: 256 -> 512
            ResidualConv1d(1024, 1024),
            ResidualUpsampling1d(in_channels=1024, out_channels=1024),  # L: 512 -> 1024
            ResidualConv1d(1024, 1024),
            ResidualConv1d(1024, 1024),
        )

        self.spatial_upsampling = nn.Sequential(
            ResidualConv2d(1024, 1024),
            ResidualHeightUpsampling2d(in_channels=1024, out_channels=512), # H: 1 -> 2
            ResidualConv2d(512, 512),
            ResidualHeightUpsampling2d(in_channels=512, out_channels=256), # H: 2 -> 4
            ResidualConv2d(256, 256),
            ResidualHeightUpsampling2d(in_channels=256, out_channels=128), # H: 4 -> 8
            ResidualConv2d(128, 128),
            ResidualHeightUpsampling2d(in_channels=128, out_channels=64),  # H: 8 -> 16
            ResidualConv2d(64, 64),
            ResidualHeightUpsampling2d(in_channels=64, out_channels=32),  # H: 16 -> 32
            ResidualConv2d(32, 32),
            ResidualHeightUpsampling2d(in_channels=32, out_channels=16),  # H: 32 -> 64
            ResidualConv2d(16, 16),
            ResidualHeightUpsampling2d(in_channels=16, out_channels=8),   # H: 64 -> 128
            ResidualConv2d(8, 8),
        )

        self.conv2d = nn.Sequential(
            ResidualConv2d(in_channels=8, out_channels=8),
            ResidualConv2d(in_channels=8, out_channels=4),
            ResidualConv2d(in_channels=4, out_channels=2),
            ResidualConv2d(in_channels=2, out_channels=1),
        )


    def forward(self, x):
        # x: B, 30, 768
        x = x.transpose(-2, -1)
        # x: B, 768, 30
        x = self.input_padding(x)
        # x: B, 768, 32

        x = self.temporal_upsampling(x)
        # x: B, 1024, 1024
        x = x.unsqueeze(-2)
        # x: B, 1024, 1, 1024
        x = self.spatial_upsampling(x)
        # x: B, 8, 128, 1024
        x = F.interpolate(x, size=(112, 799), mode='bilinear', align_corners=False)
        # x: B, 8, 112, 799
        x = self.conv2d(x)
        # x: B, 1, 112, 799
        x = torch.sigmoid(x)
        x = x.squeeze(-3).transpose(-2, -1)
        # x: B, 799, 112
        return x

if __name__ == '__main__':
    x = torch.randn((16, 30, 768))
    d = Decoder()
    print(d(x).shape)  # outputs (16, 799, 112)
