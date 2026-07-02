"""
U-Net with ResNet-34 encoder for binary skin lesion segmentation.

Architecture details:
  - Encoder : ResNet-34 (ImageNet pre-trained)
  - Decoder : Transposed-convolution upsampling with skip connections
  - Output  : Single-channel logit map (apply sigmoid to get probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Two consecutive Conv2d → BatchNorm2d → ReLU layers."""
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
    """Upsample → concatenation of skip connection → ConvBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad dynamically if spatial dims don't match exactly (due to odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """
    ResNet-34 encoder coupled with a custom U-Net decoder.
    
    Compatible with pre-trained PyTorch ResNet-34 checkpoint models.
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # Encoder layers extracted from ResNet-34
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        # Bottleneck at the lowest resolution
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder stages matching encoder resolutions
        self.dec4 = UpBlock(1024, 512, 512)
        self.dec3 = UpBlock(512,  256, 256)
        self.dec2 = UpBlock(256,  128, 128)
        self.dec1 = UpBlock(128,   64,  64)

        # Legacy/compatibility layers: must match names in original 'best_model.pth' checkpoint
        self.up_to_half  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.skip_conv   = ConvBlock(64 + 64, 64)
        self.up_final    = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv  = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        e0 = self.enc0(x)
        p  = self.pool(e0)
        e1 = self.enc1(p)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck forward pass
        b  = self.bottleneck(e4)

        # Decoder forward pass with skip connections
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Align shapes if they differ
        if d1.shape[2:] != e0.shape[2:]:
            d1 = F.pad(d1, [0, e0.shape[3] - d1.shape[3],
                            0, e0.shape[2] - d1.shape[2]])
        
        # Merge with high-resolution skip connection enc0
        merged = self.skip_conv(torch.cat([d1, e0], dim=1))
        up     = self.up_final(merged)
        return self.final_conv(up)


if __name__ == "__main__":
    # Sanity check validation run
    with torch.no_grad():
        m   = UNet(pretrained=False)
        test_x = torch.randn(2, 3, 256, 256)
        out = m(test_x)
        assert out.shape == (2, 1, 256, 256), f"Unexpected shape: {out.shape}"
        print(f"✅ UNet output shape : {out.shape}")
        total = sum(p.numel() for p in m.parameters())
        print(f"✅ Total parameters  : {total / 1e6:.2f}M")