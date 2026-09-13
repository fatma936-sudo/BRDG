"""ResNet encoder + UNet-style decoder producing the shared feature map F."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

from brdg.config import DEFAULT_FEAT_CH

# name -> (constructor, ImageNet weights, encoder stage channels)
BACKBONES = {
    "resnet34":  (resnet34,  ResNet34_Weights.IMAGENET1K_V1,  (64, 64, 128, 256, 512)),
    "resnet50":  (resnet50,  ResNet50_Weights.IMAGENET1K_V1,  (64, 256, 512, 1024, 2048)),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1, (64, 256, 512, 1024, 2048)),
}


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsample -> 1x1 reduce -> concat skip -> DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.block = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResNetUNetBackbone(nn.Module):
    """
    Encoder: ResNet-34 / 50 / 101 (ImageNet pretrained), stages at strides
    {1/2, 1/4, 1/8, 1/16, 1/32}.
    Decoder: UNet-style with four lateral skips.
    Output: the shared feature map F of shape (B, feat_ch, H, W).
    """

    def __init__(self, arch: str = "resnet34", feat_ch: int = DEFAULT_FEAT_CH,
                 pretrained: bool = True, freeze_bn: bool = True):
        super().__init__()
        if arch not in BACKBONES:
            raise ValueError(f"unknown backbone {arch!r}; choose from {sorted(BACKBONES)}")
        self.arch = arch
        self.freeze_bn = bool(freeze_bn)

        ctor, weights, chans = BACKBONES[arch]
        enc = ctor(weights=weights if pretrained else None)
        self.enc = nn.ModuleDict({
            "conv1": enc.conv1, "bn1": enc.bn1, "relu": enc.relu, "maxpool": enc.maxpool,
            "layer1": enc.layer1, "layer2": enc.layer2,
            "layer3": enc.layer3, "layer4": enc.layer4,
        })

        c1, c2, c3, c4, c5 = chans
        self.up4 = Up(c5, c4, 512)    # 1/16
        self.up3 = Up(512, c3, 256)   # 1/8
        self.up2 = Up(256, c2, 128)   # 1/4
        self.up1 = Up(128, c1, 64)    # 1/2

        self.out_conv = nn.Sequential(
            DoubleConv(64, 64),
            nn.Conv2d(64, feat_ch, kernel_size=1, bias=True),
        )

        if self.freeze_bn:
            for m in self.enc.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = False
            self._set_encoder_bn_eval()

    def _set_encoder_bn_eval(self) -> None:
        for m in self.enc.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def train(self, mode: bool = True):
        # nn.Module.train() recurses into children, so encoder BatchNorms must be
        # put back into eval mode after every call or their running statistics
        # resume updating despite freeze_bn.
        super().train(mode)
        if self.freeze_bn:
            self._set_encoder_bn_eval()
        return self

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        x = self.enc["conv1"](x)
        x = self.enc["bn1"](x)
        x = self.enc["relu"](x); s1 = x
        x = self.enc["maxpool"](x)
        x = self.enc["layer1"](x); s2 = x
        x = self.enc["layer2"](x); s3 = x
        x = self.enc["layer3"](x); s4 = x
        x = self.enc["layer4"](x)
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.out_conv(x)
