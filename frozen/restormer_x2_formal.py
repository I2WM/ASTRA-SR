from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import starir_x2_formal as formal


class RestormerX2(nn.Module):
    """Frozen historical Restormer-small architecture with a shared x2 head."""

    def __init__(self, source_root: Path) -> None:
        super().__init__()
        sys.path.insert(0, str(source_root))
        from baseline.Restormer.code.restormer import RestormerNet

        self.net = RestormerNet(
            inp_channels=1,
            out_channels=1,
            dim=16,
            num_blocks=(1, 1, 1, 2),
            num_refinement_blocks=1,
            heads=(1, 2, 4, 8),
            ffn_expansion_factor=2.0,
            bias=False,
        )
        self.net.output = nn.Conv2d(32, 4, 3, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        nn.init.zeros_(self.net.output.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.net.encoder_level1(self.net.patch_embed(x))
        x2 = self.net.encoder_level2(self.net.down1_2(x1))
        x3 = self.net.encoder_level3(self.net.down2_3(x2))
        features = self.net.latent(self.net.down3_4(x3))
        features = self.net.up4_3(features)
        features = self.net.reduce_chan_level3(torch.cat((features, x3), dim=1))
        features = self.net.decoder_level3(features)
        features = self.net.up3_2(features)
        features = self.net.reduce_chan_level2(torch.cat((features, x2), dim=1))
        features = self.net.decoder_level2(features)
        features = self.net.up2_1(features)
        features = self.net.decoder_level1(torch.cat((features, x1), dim=1))
        features = self.net.refinement(features)
        return base + self.shuffle(self.net.output(features))


def model_contract(_: str) -> dict[str, str]:
    wrapper = Path(__file__).resolve()
    return {
        "family": "Restormer",
        "configuration": "dim16_blocks1-1-1-2_heads1-2-4-8_refine1_ffn2",
        "x2_head": "zero_init_conv3x3_to4_pixelshuffle2_residual_over_bilinear",
        "wrapper_source": str(wrapper),
        "wrapper_sha256": formal.sha256(wrapper),
    }


def build_model(_: str, source_root: Path) -> nn.Module:
    return RestormerX2(source_root)


if __name__ == "__main__":
    # The shared runner's CLI family token remains internal; both factory and
    # contract are replaced before argument parsing and checkpoint creation.
    formal.build_x2_model = build_model
    formal.x2_model_contract = model_contract
    formal.main()
