import torch
import torch.nn as nn
import torch.nn.functional as F

from d4_transforms import D4, D4_INV


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BaselineUNet(nn.Module):
    """Small UNet without any symmetry constraints."""

    def __init__(self, in_channels=3, base_ch=16):
        super().__init__()
        ch = base_ch
        self.enc1 = _ConvBlock(in_channels, ch)
        self.enc2 = _ConvBlock(ch, ch * 2)
        self.enc3 = _ConvBlock(ch * 2, ch * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(ch * 4, ch * 8)

        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = _ConvBlock(ch * 8, ch * 4)

        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = _ConvBlock(ch * 4, ch * 2)

        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = _ConvBlock(ch * 2, ch)

        self.out_conv = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# orbit averaging: f_G(x) = (1/|G|) * sum_{g in G} g^{-1}( f(g(x)) )
class EquivariantUNet(nn.Module):

    def __init__(self, in_channels=3, base_ch=16):
        super().__init__()
        self.base = BaselineUNet(in_channels=in_channels, base_ch=base_ch)

    def forward(self, x):
        acc = torch.zeros_like(x[:, :1])   # B×1×H×W accumulator
        for g, g_inv in zip(D4, D4_INV):
            gx = g(x)
            f_gx = self.base(gx)
            acc = acc + g_inv(f_gx)
        return acc / len(D4)
