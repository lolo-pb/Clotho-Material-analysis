import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1))


class FiberglassUNet(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v2(weights=weights).features[:-1]
        self.decode16 = DecoderBlock(320, 96, 192)
        self.decode8 = DecoderBlock(192, 32, 96)
        self.decode4 = DecoderBlock(96, 24, 64)
        self.decode2 = DecoderBlock(64, 16, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        skips: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in (1, 3, 6, 13):
                skips[index] = x

        x = self.decode16(x, skips[13])
        x = self.decode8(x, skips[6])
        x = self.decode4(x, skips[3])
        x = self.decode2(x, skips[1])
        x = self.head(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
