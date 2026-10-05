from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from starir_x2_speed_smoke import StarIRX2


def _zero_module(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class NAFNetX2(nn.Module):
    def __init__(self, source_root: Path) -> None:
        super().__init__()
        sys.path.insert(0, str(source_root))
        from baseline.NAFNet.code.nafnet import NAFNetRestorationNet

        self.net = NAFNetRestorationNet(
            input_channels=1,
            output_channels=1,
            width=32,
            encoder_blocks=(1, 1, 2),
            middle_blocks=2,
            decoder_blocks=(1, 1, 1),
            dw_expand=2.0,
            ffn_expand=2.0,
        )
        self.net.ending = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.shuffle = nn.PixelShuffle(2)
        _zero_module(self.net.ending)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.net.encoder1(self.net.intro(x))
        x2 = self.net.encoder2(self.net.down1(x1))
        x3 = self.net.encoder3(self.net.down2(x2))
        features = self.net.middle(self.net.down3(x3))
        features = self.net.decoder3(self.net.up3(features) + x3)
        features = self.net.decoder2(self.net.up2(features) + x2)
        features = self.net.decoder1(self.net.up1(features) + x1)
        return base + self.shuffle(self.net.ending(features))


class FFTformerX2(nn.Module):
    def __init__(self, source_root: Path) -> None:
        super().__init__()
        sys.path.insert(0, str(source_root))
        from baseline.FFTformer.code.fftformer import FFTformerRestorationNet

        self.net = FFTformerRestorationNet(
            input_channels=1,
            output_channels=1,
            dim=24,
            num_blocks=(1, 2, 2, 2),
            num_refinement_blocks=1,
            ffn_expansion_factor=2.0,
            bias=False,
        )
        self.net.output = nn.Conv2d(48, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        _zero_module(self.net.output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.net.encoder_level1(self.net.patch_embed(x))
        x2 = self.net.encoder_level2(self.net.down1_2(x1))
        x3 = self.net.encoder_level3(self.net.down2_3(x2))
        features = self.net.latent(self.net.down3_4(x3))
        features = self.net.up4_3(features)
        features = self.net.reduce_chan_level3(torch.cat([features, x3], dim=1))
        features = self.net.decoder_level3(features)
        features = self.net.up3_2(features)
        features = self.net.reduce_chan_level2(torch.cat([features, x2], dim=1))
        features = self.net.decoder_level2(features)
        features = self.net.up2_1(features)
        features = self.net.decoder_level1(torch.cat([features, x1], dim=1))
        features = self.net.refinement(features)
        return base + self.shuffle(self.net.output(features))


class ConvIRX2(nn.Module):
    def __init__(self, source_root: Path) -> None:
        super().__init__()
        sys.path.insert(0, str(source_root))
        from baseline.ConvIR.code.convir import ConvIRBasicConv, ConvIRRestorationNet

        self.net = ConvIRRestorationNet(
            input_channels=1,
            output_channels=1,
            base_channel=24,
            num_res=6,
        )
        self.net.convs_out[0] = ConvIRBasicConv(96, 4, kernel_size=3, stride=1, relu=False)
        self.net.convs_out[1] = ConvIRBasicConv(48, 4, kernel_size=3, stride=1, relu=False)
        self.net.feat_extract[5] = ConvIRBasicConv(24, 4, kernel_size=3, stride=1, relu=False)
        self.shuffle = nn.PixelShuffle(2)
        _zero_module(self.net.convs_out[0])
        _zero_module(self.net.convs_out[1])
        _zero_module(self.net.feat_extract[5])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x_half = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x_quarter = F.interpolate(x_half, scale_factor=0.5, mode="bilinear", align_corners=False)
        z_half = self.net.scm2(x_half)
        z_quarter = self.net.scm1(x_quarter)
        x0 = self.net.feat_extract[0](x)
        res1 = self.net.encoder[0](x0)
        z = self.net.feat_extract[1](res1)
        z = self.net.fam2(z, z_half)
        res2 = self.net.encoder[1](z)
        z = self.net.feat_extract[2](res2)
        z = self.net.fam1(z, z_quarter)
        z = self.net.encoder[2](z)
        z = self.net.decoder[0](z)
        out_quarter = self.shuffle(self.net.convs_out[0](z)) + F.interpolate(
            x_quarter, scale_factor=2, mode="bilinear", align_corners=False
        )
        z = self.net.feat_extract[3](z)
        z = self.net.convs[0](torch.cat([z, res2], dim=1))
        z = self.net.decoder[1](z)
        out_half = self.shuffle(self.net.convs_out[1](z)) + F.interpolate(
            x_half, scale_factor=2, mode="bilinear", align_corners=False
        )
        z = self.net.feat_extract[4](z)
        z = self.net.convs[1](torch.cat([z, res1], dim=1))
        z = self.net.decoder[2](z)
        out_full = self.shuffle(self.net.feat_extract[5](z)) + F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        return [out_quarter, out_half, out_full]


def build_x2_model(family: str, source_root: Path) -> nn.Module:
    family = str(family).strip().lower()
    if family == "starir":
        return StarIRX2(source_root)
    if family == "nafnet":
        return NAFNetX2(source_root)
    if family == "fftformer":
        return FFTformerX2(source_root)
    if family == "convir":
        return ConvIRX2(source_root)
    raise ValueError(f"Unsupported x2 baseline family: {family}")


def primary_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        return output[-1]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    return output


def output_list(output: Any) -> list[torch.Tensor]:
    return list(output) if isinstance(output, (list, tuple)) else [output]


def x2_model_contract(family: str) -> dict[str, Any]:
    family = str(family).strip().lower()
    contracts = {
        "starir": {"family": "StarIR", "configuration": "dim24_blocks1-2-2_refine1_ffn3"},
        "nafnet": {"family": "NAFNet", "configuration": "width32_enc1-1-2_mid2_dec1-1-1"},
        "fftformer": {"family": "FFTformer", "configuration": "dim24_blocks1-2-2-2_refine1_ffn2"},
        "convir": {"family": "ConvIR", "configuration": "base24_numres6_three_scale_deep_supervision"},
    }
    if family not in contracts:
        raise ValueError(f"Unsupported x2 baseline family: {family}")
    return {
        **contracts[family],
        "x2_head": "zero_init_conv3x3_to4_pixelshuffle2_residual_over_bilinear",
    }
