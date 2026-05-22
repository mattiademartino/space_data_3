import torch
import torch.nn as nn
from typing import Sequence


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResDoubleConv(nn.Module):
    """Double conv with a residual (skip) connection."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))


class UNet(nn.Module):
    """Standard 3-stage U-Net (same as homework 2 baseline)."""
    def __init__(self, features: Sequence[int] = (32, 64, 128), dropout: float = 0.0):
        super().__init__()
        f = features
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1, f[0], dropout)
        self.enc2 = DoubleConv(f[0], f[1], dropout)
        self.enc3 = DoubleConv(f[1], f[2], dropout)
        self.bottleneck = DoubleConv(f[2], f[2] * 2, dropout)
        self.up3 = nn.ConvTranspose2d(f[2] * 2, f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2] * 2, f[2], dropout)
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1] * 2, f[1], dropout)
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0] * 2, f[0], dropout)
        self.out_conv = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


class UNetDeep(nn.Module):
    """4-stage U-Net; bottleneck at 4×4 for 64×64 inputs."""
    def __init__(self, features: Sequence[int] = (32, 64, 128, 256), dropout: float = 0.0):
        super().__init__()
        f = features
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1, f[0], dropout)
        self.enc2 = DoubleConv(f[0], f[1], dropout)
        self.enc3 = DoubleConv(f[1], f[2], dropout)
        self.enc4 = DoubleConv(f[2], f[3], dropout)
        self.bottleneck = DoubleConv(f[3], f[3] * 2, dropout)
        self.up4 = nn.ConvTranspose2d(f[3] * 2, f[3], 2, stride=2)
        self.dec4 = DoubleConv(f[3] * 2, f[3], dropout)
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2] * 2, f[2], dropout)
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1] * 2, f[1], dropout)
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0] * 2, f[0], dropout)
        self.out_conv = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


class ResUNet(nn.Module):
    """3-stage U-Net with residual double-conv blocks."""
    def __init__(self, features: Sequence[int] = (32, 64, 128), dropout: float = 0.0):
        super().__init__()
        f = features
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResDoubleConv(1, f[0], dropout)
        self.enc2 = ResDoubleConv(f[0], f[1], dropout)
        self.enc3 = ResDoubleConv(f[1], f[2], dropout)
        self.bottleneck = ResDoubleConv(f[2], f[2] * 2, dropout)
        self.up3 = nn.ConvTranspose2d(f[2] * 2, f[2], 2, stride=2)
        self.dec3 = ResDoubleConv(f[2] * 2, f[2], dropout)
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = ResDoubleConv(f[1] * 2, f[1], dropout)
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = ResDoubleConv(f[0] * 2, f[0], dropout)
        self.out_conv = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


class Autoencoder(nn.Module):
    """Symmetric CNN encoder-decoder without skip connections."""
    def __init__(self, features: Sequence[int] = (32, 64, 128), dropout: float = 0.0):
        super().__init__()
        f = features
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1, f[0], dropout)
        self.enc2 = DoubleConv(f[0], f[1], dropout)
        self.enc3 = DoubleConv(f[1], f[2], dropout)
        self.bottleneck = DoubleConv(f[2], f[2] * 2, dropout)
        self.up3 = nn.ConvTranspose2d(f[2] * 2, f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2], f[2], dropout)
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1], f[1], dropout)
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0], f[0], dropout)
        self.out_conv = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self.up3(b))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))
        return torch.sigmoid(self.out_conv(d1))


class ResBlock(nn.Module):
    """Single residual block: Conv-BN-ReLU-Conv-BN + identity skip."""
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNetDenoiser(nn.Module):
    """
    Flat ResNet denoiser — full resolution throughout (no pooling/upsampling).
    Processes the image with stacked residual blocks, keeping spatial dims fixed.
    Unlike U-Net, context is captured through receptive field growth, not skip connections.
    """
    def __init__(self, base_channels: int = 64, n_blocks: int = 8, dropout: float = 0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(base_channels, dropout) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.tail(self.body(self.head(x))))


def build_model(model_cfg: dict) -> nn.Module:
    arch = model_cfg["architecture"]
    features = list(model_cfg.get("features", [32, 64, 128]))
    dropout = float(model_cfg.get("dropout", 0.0))

    if arch == "unet":
        return UNet(features, dropout)
    elif arch == "unet_deep":
        return UNetDeep(features, dropout)
    elif arch == "res_unet":
        return ResUNet(features, dropout)
    elif arch == "autoencoder":
        return Autoencoder(features, dropout)
    elif arch == "resnet":
        base_channels = int(model_cfg.get("base_channels", 64))
        n_blocks = int(model_cfg.get("n_blocks", 8))
        return ResNetDenoiser(base_channels, n_blocks, dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}")
