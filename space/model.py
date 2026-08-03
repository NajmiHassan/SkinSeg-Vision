"""
U-Net with a ResNet-34 encoder for binary skin lesion segmentation.

  Encoder : ResNet-34, ImageNet pre-trained
  Decoder : transposed-convolution upsampling with skip connections
  Output  : single-channel logit map (apply sigmoid for probabilities)

Resolution ladder for a 256x256 input:

    enc0   64ch   128x128     <- skip
    pool          64x64
    enc1   64ch   64x64       <- skip
    enc2  128ch   32x32       <- skip
    enc3  256ch   16x16       <- skip
    enc4  512ch    8x8        <- skip
    bottleneck 1024ch 8x8
    dec4 -> dec3 -> dec2 -> dec1 back up to 128x128
    skip_conv merges dec1 with enc0, up_final lifts to 256x256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Two consecutive Conv2d -> BatchNorm2d -> ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample -> concatenate skip connection -> ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad when spatial dims disagree, which happens for non-power-of-two inputs.
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """ResNet-34 encoder coupled with a custom U-Net decoder."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # Encoder taken from ResNet-34
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.bottleneck = ConvBlock(512, 1024)

        self.dec4 = UpBlock(1024, 512, 512)
        self.dec3 = UpBlock(512, 256, 256)
        self.dec2 = UpBlock(256, 128, 128)
        self.dec1 = UpBlock(128, 64, 64)

        # NOTE: up_to_half is never called in forward(). It is a leftover from
        # an earlier decoder design. It stays declared because the released
        # checkpoint contains up_to_half.{weight,bias} and removing the layer
        # would break a strict load_state_dict(). It costs ~16k dead parameters
        # and zero compute. Drop it together with the next retrain, not before.
        self.up_to_half = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.skip_conv = ConvBlock(64 + 64, 64)
        self.up_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        p = self.pool(e0)
        e1 = self.enc1(p)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        if d1.shape[2:] != e0.shape[2:]:
            d1 = F.pad(d1, [0, e0.shape[3] - d1.shape[3],
                            0, e0.shape[2] - d1.shape[2]])

        merged = self.skip_conv(torch.cat([d1, e0], dim=1))
        up = self.up_final(merged)
        return self.final_conv(up)


if __name__ == "__main__":
    with torch.no_grad():
        m = UNet(pretrained=False)
        out = m(torch.randn(2, 3, 256, 256))
        assert out.shape == (2, 1, 256, 256), f"Unexpected shape: {out.shape}"
        print(f"UNet output shape : {tuple(out.shape)}")
        print(f"Total parameters  : {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
