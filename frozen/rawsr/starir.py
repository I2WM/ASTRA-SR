from __future__ import annotations

import numbers
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
except Exception:  # pragma: no cover - torch/einops are optional in this workspace
    torch = None
    nn = None
    F = None
    rearrange = None


class _MissingTorchModule:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("starir.py requires working torch and einops installations")


ModuleBase = nn.Module if nn is not None else _MissingTorchModule


def _require_torch() -> None:
    if torch is None or nn is None or F is None or rearrange is None:
        raise RuntimeError("starir.py requires working torch and einops installations")


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_4d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class StarIRBiasFreeLayerNorm(ModuleBase):
    def __init__(self, normalized_shape: int | tuple[int, ...]) -> None:
        _require_torch()
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        shape = torch.Size(normalized_shape)
        if len(shape) != 1:
            raise ValueError("StarIRBiasFreeLayerNorm expects a 1D normalized shape")
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class StarIRWithBiasLayerNorm(ModuleBase):
    def __init__(self, normalized_shape: int | tuple[int, ...]) -> None:
        _require_torch()
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        shape = torch.Size(normalized_shape)
        if len(shape) != 1:
            raise ValueError("StarIRWithBiasLayerNorm expects a 1D normalized shape")
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class StarIRLayerNorm(ModuleBase):
    def __init__(self, dim: int, layernorm_type: str) -> None:
        _require_torch()
        super().__init__()
        if str(layernorm_type) == "BiasFree":
            self.body = StarIRBiasFreeLayerNorm(dim)
        else:
            self.body = StarIRWithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), height, width)


class StarIRSpatialOperation(ModuleBase):
    def __init__(self, dim: int) -> None:
        _require_torch()
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class StarIRDFFN(ModuleBase):
    def __init__(
        self,
        dim: int,
        *,
        ffn_expansion_factor: float,
        bias: bool,
        patch_size: int = 8,
    ) -> None:
        _require_torch()
        super().__init__()
        hidden_features = int(dim * float(ffn_expansion_factor))
        self.patch_size = int(patch_size)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.fft_weight = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)

        patch = rearrange(
            x,
            "b c (h ph) (w pw) -> b c h w ph pw",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        patch_fft = torch.fft.rfft2(patch.float())
        patch = torch.fft.irfft2(patch_fft * self.fft_weight, s=(self.patch_size, self.patch_size))
        return rearrange(
            patch,
            "b c h w ph pw -> b c (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )


class StarIRModule(ModuleBase):
    def __init__(self, dim: int, *, bias: bool, patch_size: int = 8) -> None:
        _require_torch()
        super().__init__()
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.to_hidden = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm = StarIRLayerNorm(dim, layernorm_type="WithBias")
        self.fft_weight = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.spatial = StarIRSpatialOperation(dim)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.to_hidden(x)
        q, v = self.to_hidden_dw(hidden).split([self.dim, self.dim], dim=1)

        q_patch = rearrange(
            q,
            "b c (h ph) (w pw) -> b c h w ph pw",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        q_fft = torch.fft.rfft2(q_patch.float())
        out = torch.fft.irfft2(q_fft * self.fft_weight, s=(self.patch_size, self.patch_size))
        out = rearrange(
            out,
            "b c h w ph pw -> b c (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        out = self.norm(out)
        out = self.spatial(v) * out
        out = self.channel_gate(out) * out
        return self.project_out(out)


class StarIRBlock(ModuleBase):
    def __init__(
        self,
        dim: int,
        *,
        ffn_expansion_factor: float = 3.0,
        bias: bool = False,
        layernorm_type: str = "WithBias",
    ) -> None:
        _require_torch()
        super().__init__()
        self.norm1 = StarIRLayerNorm(dim, layernorm_type)
        self.attn = StarIRModule(dim, bias=bias)
        self.norm2 = StarIRLayerNorm(dim, layernorm_type)
        self.ffn = StarIRDFFN(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StarIRFuse(ModuleBase):
    def __init__(self, channels: int) -> None:
        _require_torch()
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat((encoder, decoder), dim=1))


class StarIROverlapPatchEmbed(ModuleBase):
    def __init__(self, in_channels: int, embed_dim: int, *, bias: bool = False) -> None:
        _require_torch()
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class StarIRDownsample(ModuleBase):
    def __init__(self, channels: int) -> None:
        _require_torch()
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StarIRUpsample(ModuleBase):
    def __init__(self, channels: int) -> None:
        _require_torch()
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StarIRBandResidualAdapter(ModuleBase):
    def __init__(self, channels: int, low_cut: float = 0.2, mid_cut: float = 0.5) -> None:
        _require_torch()
        super().__init__()
        if not 0.0 < float(low_cut) < float(mid_cut) < 1.0:
            raise ValueError("band cuts must satisfy 0 < low < mid < 1")
        self.channels = int(channels)
        self.low_cut = float(low_cut)
        self.mid_cut = float(mid_cut)
        self.transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=1,
                        groups=self.channels,
                        bias=False,
                    ),
                    nn.GELU(),
                    nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
                )
                for _ in range(3)
            ]
        )
        self.fuse = nn.Conv2d(self.channels * 3, self.channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fuse.weight)

    @staticmethod
    def radial_masks(
        height: int,
        width: int,
        *,
        device: torch.device,
        low_cut: float = 0.2,
        mid_cut: float = 0.5,
    ) -> tuple[torch.Tensor, ...]:
        fy = torch.fft.fftfreq(int(height), device=device).view(-1, 1)
        fx = torch.fft.rfftfreq(int(width), device=device).view(1, -1)
        radius = torch.sqrt(fy.square() + fx.square()) / (0.5 * (2.0**0.5))
        return (
            (radius < low_cut).float(),
            ((radius >= low_cut) & (radius < mid_cut)).float(),
            (radius >= mid_cut).float(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm="ortho")
            masks = self.radial_masks(
                x.shape[-2],
                x.shape[-1],
                device=x.device,
                low_cut=self.low_cut,
                mid_cut=self.mid_cut,
            )
            bands = [
                torch.fft.irfft2(spectrum * mask, s=x.shape[-2:], norm="ortho")
                for mask in masks
            ]
        transformed = [
            module(band.to(dtype=x.dtype))
            for module, band in zip(self.transforms, bands)
        ]
        return x + self.fuse(torch.cat(transformed, dim=1))


class StarIRNAFRefinement(ModuleBase):
    """A zero-initialized NAF-style local residual correction block."""

    def __init__(self, channels: int, *, layernorm_type: str = "WithBias") -> None:
        _require_torch()
        super().__init__()
        hidden_channels = int(channels) * 2
        self.norm1 = StarIRLayerNorm(int(channels), layernorm_type)
        self.expand = nn.Conv2d(int(channels), hidden_channels, kernel_size=1, bias=True)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=True,
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=1, bias=True),
        )
        self.project = nn.Conv2d(hidden_channels // 2, int(channels), kernel_size=1, bias=True)
        self.norm2 = StarIRLayerNorm(int(channels), layernorm_type)
        self.ffn_expand = nn.Conv2d(int(channels), hidden_channels, kernel_size=1, bias=True)
        self.ffn_project = nn.Conv2d(hidden_channels // 2, int(channels), kernel_size=1, bias=True)
        # Start as the identity so B and C share the same primary initialization.
        self.beta = nn.Parameter(torch.zeros(1, int(channels), 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, int(channels), 1, 1))

    @staticmethod
    def _simple_gate(x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm1(x)
        residual = self._simple_gate(self.depthwise(self.expand(residual)))
        residual = residual * self.channel_attention(residual)
        x = x + self.beta * self.project(residual)
        residual = self._simple_gate(self.ffn_expand(self.norm2(x)))
        return x + self.gamma * self.ffn_project(residual)


class StarIRRestorationNet(ModuleBase):
    def __init__(
        self,
        *,
        input_channels: int = 1,
        output_channels: int = 1,
        dim: int = 48,
        num_blocks: tuple[int, int, int] = (2, 3, 4),
        num_refinement_blocks: int = 2,
        ffn_expansion_factor: float = 3.0,
        bias: bool = False,
        dual_pixel_task: bool = False,
        use_band_adapter: bool = False,
        band_low_cut: float = 0.2,
        band_mid_cut: float = 0.5,
        use_ofr: bool = False,
        ofr_prompt_channels: int = 32,
        ofr_expert_channels: int = 24,
        ofr_window_size: int = 8,
        ofr_temperature: float = 1.0,
        ofr_residual_scale: float = 1.0,
          use_source_conditioning: bool = False,
          source_residual_scale: float = 1.0,
          source_residual_target: str = "both",
          use_enc3_aux: bool = False,
          use_enc3_blur_aux: bool = False,
          use_naf_refinement: bool = False,
          naf_refinement_blocks: int = 1,
      ) -> None:
        _require_torch()
        super().__init__()
        if len(tuple(num_blocks)) != 3:
            raise ValueError("num_blocks must contain exactly three stage depths")
        self.patch_embed = StarIROverlapPatchEmbed(input_channels, dim, bias=bias)
        self.encoder_level1 = nn.Sequential(
            *[
                StarIRBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(int(num_blocks[0]))
            ]
        )
        self.down1_2 = StarIRDownsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                StarIRBlock(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(int(num_blocks[1]))
            ]
        )
        self.down2_3 = StarIRDownsample(dim * 2)
        self.encoder_level3 = nn.Sequential(
            *[
                StarIRBlock(dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(int(num_blocks[2]))
            ]
        )
        self.decoder_level3 = nn.Sequential(
            *[
                StarIRBlock(dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(int(num_blocks[2]))
            ]
        )
        self.up3_2 = StarIRUpsample(dim * 4)
        self.fuse2 = StarIRFuse(dim * 2)
        self.decoder_level2 = nn.Sequential(
            *[
                StarIRBlock(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(int(num_blocks[1]))
            ]
        )
        self.up2_1 = StarIRUpsample(dim * 2)
        self.fuse1 = StarIRFuse(dim)
        self.decoder_level1 = nn.Sequential(
            *[
                StarIRBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(int(num_blocks[0]))
            ]
        )
        self.refinement = nn.Sequential(
            *[
                StarIRBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(int(num_refinement_blocks))
            ]
        )
        self.output = nn.Conv2d(dim, output_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        if int(input_channels) == int(output_channels):
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=True)
            with torch.no_grad():
                self.residual_projection.weight.zero_()
                self.residual_projection.bias.zero_()
                shared_channels = min(int(input_channels), int(output_channels))
                for channel_idx in range(shared_channels):
                    self.residual_projection.weight[channel_idx, channel_idx, 0, 0] = 1.0
        self.use_source_conditioning = bool(use_source_conditioning)
        self.source_residual_scale = float(source_residual_scale)
        self.source_residual_target = str(source_residual_target).strip().lower()
        if self.source_residual_target not in {"both", "real", "png"}:
            raise ValueError("source_residual_target must be one of: both, real, png")
        self.supports_source_conditioning = bool(use_source_conditioning)
        self.source_residual_head = None
        if self.use_source_conditioning:
            self.source_residual_head = nn.Conv2d(
                dim,
                int(output_channels) * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            nn.init.zeros_(self.source_residual_head.weight)
            nn.init.zeros_(self.source_residual_head.bias)
        self.dual_pixel_task = bool(dual_pixel_task)
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.use_ofr = bool(use_ofr)
        self.ofr_refiner = None
        if self.use_ofr:
            if int(input_channels) != int(output_channels):
                raise ValueError("OFR requires input_channels == output_channels")
            from .ofr import OFRConfig, ObjectFrequencyRoutedRefiner

            self.ofr_refiner = ObjectFrequencyRoutedRefiner(
                OFRConfig(
                    image_channels=int(input_channels),
                    level_channels=(int(dim), int(dim) * 2, int(dim) * 4),
                    prompt_channels=int(ofr_prompt_channels),
                    expert_channels=int(ofr_expert_channels),
                    window_size=int(ofr_window_size),
                    temperature=float(ofr_temperature),
                    residual_scale=float(ofr_residual_scale),
                )
            )
        if bool(use_enc3_aux) and bool(use_enc3_blur_aux):
            raise ValueError("only one StarIR encoder auxiliary head can be enabled")
        if bool(use_enc3_blur_aux) and int(input_channels) < int(output_channels):
            raise ValueError(
                "enc3 blur auxiliary requires input_channels >= output_channels"
            )
        self.output_channels = int(output_channels)
        self.use_enc3_aux = bool(use_enc3_aux)
        self.supports_enc3_auxiliary = self.use_enc3_aux
        self.enc3_aux_head = None
        if self.use_enc3_aux:
            # Create this after the primary path so paired seeds retain identical
            # initialization for every inference-time parameter.
            self.enc3_aux_head = nn.Conv2d(
                dim * 4,
                int(output_channels),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        self.use_enc3_blur_aux = bool(use_enc3_blur_aux)
        self.supports_enc3_blur_auxiliary = self.use_enc3_blur_aux
        self.enc3_blur_aux_head = None
        if self.use_enc3_blur_aux:
            # The encoder predicts only the noise correction. The downsampled
            # observed image remains the explicit PSF-blurred base estimate.
            self.enc3_blur_aux_head = nn.Conv2d(
                dim * 4,
                self.output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        # Construct optional modules only after every shared inference parameter.
        # This keeps identical seeds comparable to the no-adapter baseline.
        self.band_adapter = (
            StarIRBandResidualAdapter(
                dim * 4,
                low_cut=band_low_cut,
                mid_cut=band_mid_cut,
            )
            if use_band_adapter
            else nn.Identity()
        )
        if int(naf_refinement_blocks) <= 0:
            raise ValueError("naf_refinement_blocks must be positive")
        self.naf_refinement = (
            nn.Sequential(
                *[StarIRNAFRefinement(dim) for _ in range(int(naf_refinement_blocks))]
            )
            if use_naf_refinement
            else nn.Identity()
        )

    def _source_indices(
        self,
        source_kind: list[str] | tuple[str, ...] | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if source_kind is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if torch.is_tensor(source_kind):
            return source_kind.to(device=device, dtype=torch.long).view(-1)
        indices = [1 if str(item).lower() == "png" else 0 for item in source_kind]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(
        self,
        x: torch.Tensor,
        source_kind: list[str] | tuple[str, ...] | torch.Tensor | None = None,
        return_enc3_aux: bool = False,
        return_enc3_blur_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_enc3_aux and not self.supports_enc3_auxiliary:
            raise ValueError("enc3 auxiliary output was requested but is not configured")
        if return_enc3_blur_aux and not self.supports_enc3_blur_auxiliary:
            raise ValueError("enc3 blur auxiliary output was requested but is not configured")
        if return_enc3_aux and return_enc3_blur_aux:
            raise ValueError("only one StarIR encoder auxiliary output can be requested")
        if x.shape[-2] % 32 != 0 or x.shape[-1] % 32 != 0:
            raise ValueError("StarIRRestorationNet expects height and width divisible by 32")
        enc1 = self.patch_embed(x)
        enc1 = self.encoder_level1(enc1)
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3_raw = self.encoder_level3(self.down2_3(enc2))
        enc3_blur_aux_prediction = None
        if return_enc3_blur_aux and self.enc3_blur_aux_head is not None:
            observed_lowres = F.interpolate(
                x[:, : self.output_channels],
                size=enc3_raw.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            enc3_blur_aux_prediction = observed_lowres + self.enc3_blur_aux_head(enc3_raw)

        enc3 = self.band_adapter(enc3_raw)
        # Preserve the legacy clean-auxiliary behavior: only the physical blur
        # target is intentionally isolated from the optional band adapter.
        enc3_aux_prediction = (
            self.enc3_aux_head(enc3)
            if return_enc3_aux and self.enc3_aux_head is not None
            else None
        )
        dec3 = self.decoder_level3(enc3)
        dec2 = self.decoder_level2(self.fuse2(enc2, self.up3_2(dec3)))
        dec1 = self.decoder_level1(self.fuse1(enc1, self.up2_1(dec2)))
        dec1 = self.refinement(dec1)
        dec1 = self.naf_refinement(dec1)

        if self.dual_pixel_task:
            dec1 = dec1 + self.skip_conv(self.patch_embed(x))
            base_output = self.output(dec1)
        else:
            base_output = self.output(dec1) + self.residual_projection(x)
        if self.use_source_conditioning and self.source_residual_head is not None:
            source_indices = self._source_indices(source_kind, x.shape[0], x.device)
            source_residual = self.source_residual_head(dec1)
            source_residual = source_residual.view(
                x.shape[0],
                2,
                -1,
                x.shape[-2],
                x.shape[-1],
            )
            gather_index = source_indices.view(-1, 1, 1, 1, 1).expand(
                -1,
                1,
                source_residual.shape[2],
                source_residual.shape[3],
                source_residual.shape[4],
            )
            selected_residual = source_residual.gather(1, gather_index).squeeze(1)
            if self.source_residual_target != "both":
                target_index = 1 if self.source_residual_target == "png" else 0
                selected_residual = torch.where(
                    (source_indices == target_index).view(-1, 1, 1, 1),
                    selected_residual,
                    torch.zeros_like(selected_residual),
                )
            base_output = base_output + self.source_residual_scale * selected_residual
        if self.use_ofr:
            base_output = base_output + self.ofr_refiner(x, enc1, enc2, enc3)
        if return_enc3_aux:
            if enc3_aux_prediction is None:
                raise RuntimeError("configured enc3 auxiliary features were not produced")
            return base_output, enc3_aux_prediction
        if return_enc3_blur_aux:
            if enc3_blur_aux_prediction is None:
                raise RuntimeError("configured enc3 blur auxiliary features were not produced")
            return base_output, enc3_blur_aux_prediction
        return base_output
