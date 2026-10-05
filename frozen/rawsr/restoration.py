from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rawsr.fits_train import FITSMaterializedTrainDataset, array_from_sample, load_fits_array

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    from torch.utils.data import Dataset as TorchDataset
except Exception:  # pragma: no cover - torch is optional in this workspace
    torch = None
    F = None
    nn = None
    activation_checkpoint = None

    class TorchDataset:  # type: ignore[no-redef]
        pass


class _MissingTorchModule:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("restoration.py requires a working torch installation")


ModuleBase = nn.Module if nn is not None else _MissingTorchModule


class SCGNChannelLayerNorm2d(ModuleBase):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        _require_torch()
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, int(channels), 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, int(channels), 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


@dataclass(frozen=True)
class SCGNModelConfig:
    model_family: str = "scgn"
    input_channels: int = 1
    output_channels: int = 1
    hidden_channels: int = 64
    num_blocks: int = 8
    branch_repeats: int = 1
    reduction: int = 8
    frequency_pad: int = 16
    fbgw_init_scale: float = 0.1
    use_sdgw: bool = True
    use_fbgw: bool = True
    global_feature_residual: bool = True
    global_output_residual: bool = True
    variant: str = "simplified"
    use_band_adapter: bool = False
    band_low_cut: float = 0.2
    band_mid_cut: float = 0.5
    use_source_conditioning: bool = False
    source_residual_scale: float = 1.0
    source_residual_target: str = "both"
    use_midpoint_aux: bool = False
    midpoint_aux_block: int = 4
    use_naf_refinement: bool = False
    naf_refinement_blocks: int = 1
    use_midband_naf_refinement: bool = False
    midband_naf_refinement_blocks: int = 1
    use_local_patch_fourier_refinement: bool = False
    local_patch_fourier_size: int = 64
    local_patch_fourier_expert_channels: int = 8
    local_patch_fourier_chunk_size: int = 16

    def validate(self) -> None:
        if infer_model_family(self) != "scgn":
            raise ValueError("SCGNModelConfig.model_family must resolve to 'scgn'")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be even so SFEB can split channels in half")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.branch_repeats <= 0:
            raise ValueError("branch_repeats must be positive")
        if self.reduction <= 0:
            raise ValueError("reduction must be positive")
        if self.frequency_pad < 0:
            raise ValueError("frequency_pad must be non-negative")
        if not np.isfinite(self.fbgw_init_scale):
            raise ValueError("fbgw_init_scale must be finite")
        if self.variant not in {"simplified", "reference", "paper", "paper_residual", "paper_residual_ln"}:
            raise ValueError("unsupported SCGN variant")
        if not 0.0 < float(self.band_low_cut) < float(self.band_mid_cut) < 1.0:
            raise ValueError("band cuts must satisfy 0 < low < mid < 1")
        if not np.isfinite(float(self.source_residual_scale)):
            raise ValueError("source_residual_scale must be finite")
        if str(self.source_residual_target).strip().lower() not in {"both", "real", "png"}:
            raise ValueError("source_residual_target must be one of: both, real, png")
        if self.use_source_conditioning and self.variant not in {"paper_residual", "paper_residual_ln"}:
            raise ValueError("SCGN source conditioning currently requires a residual paper variant")
        if self.use_midpoint_aux:
            if self.variant not in {"paper_residual", "paper_residual_ln"}:
                raise ValueError("SCGN midpoint auxiliary supervision requires a residual paper variant")
            if not 1 <= int(self.midpoint_aux_block) <= int(self.num_blocks):
                raise ValueError("midpoint_aux_block must be between 1 and num_blocks")
            if self.input_channels != self.output_channels:
                raise ValueError("SCGN midpoint residual prediction requires matching input/output channels")
        if self.naf_refinement_blocks <= 0:
            raise ValueError("SCGN naf_refinement_blocks must be positive")
        if self.midband_naf_refinement_blocks <= 0:
            raise ValueError("SCGN midband_naf_refinement_blocks must be positive")
        if int(self.local_patch_fourier_size) <= 0 or int(self.local_patch_fourier_size) % 2 != 0:
            raise ValueError("SCGN local_patch_fourier_size must be a positive even integer")
        if int(self.local_patch_fourier_expert_channels) <= 0:
            raise ValueError("SCGN local_patch_fourier_expert_channels must be positive")
        if int(self.local_patch_fourier_chunk_size) <= 0:
            raise ValueError("SCGN local_patch_fourier_chunk_size must be positive")
        if self.use_midband_naf_refinement:
            if not self.use_band_adapter or not self.use_midpoint_aux:
                raise ValueError(
                    "SCGN midband NAF refinement requires band adapter and midpoint auxiliary features"
                )
            if self.use_naf_refinement:
                raise ValueError("SCGN legacy NAF and midband NAF refinement are mutually exclusive")
        if self.use_local_patch_fourier_refinement:
            if not self.use_band_adapter or not self.use_midpoint_aux:
                raise ValueError(
                    "SCGN local patch Fourier refinement requires band adapter and midpoint auxiliary supervision"
                )
            if self.variant not in {"paper_residual", "paper_residual_ln"}:
                raise ValueError("SCGN local patch Fourier refinement requires a residual paper variant")
            if self.use_naf_refinement or self.use_midband_naf_refinement:
                raise ValueError(
                    "SCGN local patch Fourier refinement is mutually exclusive with NAF refinements"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConvIRModelConfig:
    model_family: str = "convir"
    input_channels: int = 1
    output_channels: int = 1
    base_channel: int = 32
    num_res: int = 16

    def validate(self) -> None:
        if infer_model_family(self) != "convir":
            raise ValueError("ConvIRModelConfig.model_family must resolve to 'convir'")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if self.base_channel <= 0:
            raise ValueError("base_channel must be positive")
        if self.num_res <= 0:
            raise ValueError("num_res must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StarIRModelConfig:
    model_family: str = "starir"
    input_channels: int = 1
    output_channels: int = 1
    dim: int = 48
    num_blocks_l1: int = 2
    num_blocks_l2: int = 3
    num_blocks_l3: int = 4
    num_refinement_blocks: int = 2
    ffn_expansion_factor: float = 3.0
    bias: bool = False
    use_band_adapter: bool = False
    band_low_cut: float = 0.2
    band_mid_cut: float = 0.5
    use_ofr: bool = False
    ofr_prompt_channels: int = 32
    ofr_expert_channels: int = 24
    ofr_window_size: int = 8
    ofr_temperature: float = 1.0
    ofr_residual_scale: float = 1.0
    use_source_conditioning: bool = False
    source_residual_scale: float = 1.0
    source_residual_target: str = "both"
    use_enc3_aux: bool = False
    use_enc3_blur_aux: bool = False
    use_naf_refinement: bool = False
    naf_refinement_blocks: int = 1

    def validate(self) -> None:
        if infer_model_family(self) != "starir":
            raise ValueError("StarIRModelConfig.model_family must resolve to 'starir'")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_blocks_l1 <= 0 or self.num_blocks_l2 <= 0 or self.num_blocks_l3 <= 0:
            raise ValueError("all stage block counts must be positive")
        if self.num_refinement_blocks <= 0:
            raise ValueError("num_refinement_blocks must be positive")
        if not np.isfinite(float(self.ffn_expansion_factor)) or float(self.ffn_expansion_factor) <= 0.0:
            raise ValueError("ffn_expansion_factor must be a positive finite value")
        if not 0.0 < float(self.band_low_cut) < float(self.band_mid_cut) < 1.0:
            raise ValueError("band cuts must satisfy 0 < low < mid < 1")
        if self.ofr_prompt_channels <= 0:
            raise ValueError("ofr_prompt_channels must be positive")
        if self.ofr_expert_channels <= 0:
            raise ValueError("ofr_expert_channels must be positive")
        if self.ofr_window_size <= 0:
            raise ValueError("ofr_window_size must be positive")
        if not np.isfinite(float(self.ofr_temperature)) or float(self.ofr_temperature) <= 0.0:
            raise ValueError("ofr_temperature must be a positive finite value")
        if not np.isfinite(float(self.ofr_residual_scale)):
            raise ValueError("ofr_residual_scale must be finite")
        if not np.isfinite(float(self.source_residual_scale)):
            raise ValueError("source_residual_scale must be finite")
        if str(self.source_residual_target).strip().lower() not in {"both", "real", "png"}:
            raise ValueError("source_residual_target must be one of: both, real, png")
        if self.use_ofr and self.input_channels != self.output_channels:
            raise ValueError("OFR requires input_channels == output_channels")
        if self.use_enc3_aux and self.use_enc3_blur_aux:
            raise ValueError("only one StarIR encoder auxiliary head can be enabled")
        if self.use_enc3_blur_aux and self.input_channels < self.output_channels:
            raise ValueError(
                "enc3 blur auxiliary requires input_channels >= output_channels"
            )
        if self.naf_refinement_blocks <= 0:
            raise ValueError("naf_refinement_blocks must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RestormerModelConfig:
    model_family: str = "restormer"
    input_channels: int = 1
    output_channels: int = 1
    dim: int = 32
    num_blocks_l1: int = 2
    num_blocks_l2: int = 3
    num_blocks_l3: int = 3
    num_blocks_latent: int = 4
    num_refinement_blocks: int = 2
    num_heads_l1: int = 1
    num_heads_l2: int = 2
    num_heads_l3: int = 4
    num_heads_latent: int = 8
    ffn_expansion_factor: float = 2.66
    bias: bool = False

    def validate(self) -> None:
        if infer_model_family(self) != "restormer":
            raise ValueError("RestormerModelConfig.model_family must resolve to 'restormer'")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if min(
            int(self.num_blocks_l1),
            int(self.num_blocks_l2),
            int(self.num_blocks_l3),
            int(self.num_blocks_latent),
            int(self.num_refinement_blocks),
        ) <= 0:
            raise ValueError("all Restormer stage block counts must be positive")
        if min(
            int(self.num_heads_l1),
            int(self.num_heads_l2),
            int(self.num_heads_l3),
            int(self.num_heads_latent),
        ) <= 0:
            raise ValueError("all Restormer head counts must be positive")
        if self.dim % int(self.num_heads_l1) != 0:
            raise ValueError("dim must be divisible by num_heads_l1")
        if (self.dim * 2) % int(self.num_heads_l2) != 0:
            raise ValueError("dim*2 must be divisible by num_heads_l2")
        if (self.dim * 4) % int(self.num_heads_l3) != 0:
            raise ValueError("dim*4 must be divisible by num_heads_l3")
        if (self.dim * 8) % int(self.num_heads_latent) != 0:
            raise ValueError("dim*8 must be divisible by num_heads_latent")
        if not np.isfinite(float(self.ffn_expansion_factor)) or float(self.ffn_expansion_factor) <= 0.0:
            raise ValueError("ffn_expansion_factor must be a positive finite value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MambaIRv2ModelConfig:
    model_family: str = "mambairv2"
    input_channels: int = 1
    output_channels: int = 1
    img_size: int = 1024
    patch_size: int = 1
    embed_dim: int = 48
    d_state: int = 8
    depth_l1: int = 6
    depth_l2: int = 6
    depth_l3: int = 6
    depth_l4: int = 6
    num_heads_l1: int = 4
    num_heads_l2: int = 4
    num_heads_l3: int = 4
    num_heads_l4: int = 4
    window_size: int = 16
    inner_rank: int = 32
    num_tokens: int = 64
    convffn_kernel_size: int = 5
    mlp_ratio: float = 2.0
    qkv_bias: bool = True
    patch_norm: bool = True
    use_checkpoint: bool = False
    resi_connection: str = "1conv"

    def validate(self) -> None:
        if infer_model_family(self) != "mambairv2":
            raise ValueError("MambaIRv2ModelConfig.model_family must resolve to 'mambairv2'")
        if self.input_channels <= 0 or self.output_channels <= 0:
            raise ValueError("input_channels and output_channels must be positive")
        if self.input_channels != self.output_channels:
            raise ValueError("MambaIRv2 currently expects input_channels == output_channels")
        if min(
            int(self.img_size),
            int(self.patch_size),
            int(self.embed_dim),
            int(self.d_state),
            int(self.depth_l1),
            int(self.depth_l2),
            int(self.depth_l3),
            int(self.depth_l4),
            int(self.num_heads_l1),
            int(self.num_heads_l2),
            int(self.num_heads_l3),
            int(self.num_heads_l4),
            int(self.window_size),
            int(self.inner_rank),
            int(self.num_tokens),
            int(self.convffn_kernel_size),
        ) <= 0:
            raise ValueError("all MambaIRv2 integer hyperparameters must be positive")
        if not np.isfinite(float(self.mlp_ratio)) or float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be a positive finite value")
        if self.resi_connection not in {"1conv", "3conv"}:
            raise ValueError("resi_connection must be '1conv' or '3conv'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NAFNetModelConfig:
    model_family: str = "nafnet"
    input_channels: int = 1
    output_channels: int = 1
    width: int = 32
    encoder_blocks_l1: int = 1
    encoder_blocks_l2: int = 1
    encoder_blocks_l3: int = 2
    middle_blocks: int = 2
    decoder_blocks_l1: int = 1
    decoder_blocks_l2: int = 1
    decoder_blocks_l3: int = 1
    dw_expand: float = 2.0
    ffn_expand: float = 2.0

    def validate(self) -> None:
        if infer_model_family(self) != "nafnet":
            raise ValueError("NAFNetModelConfig.model_family must resolve to 'nafnet'")
        if self.input_channels <= 0 or self.output_channels <= 0:
            raise ValueError("input_channels and output_channels must be positive")
        if min(
            int(self.width),
            int(self.encoder_blocks_l1),
            int(self.encoder_blocks_l2),
            int(self.encoder_blocks_l3),
            int(self.middle_blocks),
            int(self.decoder_blocks_l1),
            int(self.decoder_blocks_l2),
            int(self.decoder_blocks_l3),
        ) <= 0:
            raise ValueError("all NAFNet integer hyperparameters must be positive")
        if not np.isfinite(float(self.dw_expand)) or float(self.dw_expand) <= 0.0:
            raise ValueError("dw_expand must be a positive finite value")
        if not np.isfinite(float(self.ffn_expand)) or float(self.ffn_expand) <= 0.0:
            raise ValueError("ffn_expand must be a positive finite value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FFTformerModelConfig:
    model_family: str = "fftformer"
    input_channels: int = 1
    output_channels: int = 1
    dim: int = 24
    num_blocks_l1: int = 1
    num_blocks_l2: int = 2
    num_blocks_l3: int = 2
    num_blocks_latent: int = 2
    num_refinement_blocks: int = 1
    ffn_expansion_factor: float = 2.0
    bias: bool = False

    def validate(self) -> None:
        if infer_model_family(self) != "fftformer":
            raise ValueError("FFTformerModelConfig.model_family must resolve to 'fftformer'")
        if self.input_channels <= 0 or self.output_channels <= 0:
            raise ValueError("input_channels and output_channels must be positive")
        if min(
            int(self.dim),
            int(self.num_blocks_l1),
            int(self.num_blocks_l2),
            int(self.num_blocks_l3),
            int(self.num_blocks_latent),
            int(self.num_refinement_blocks),
        ) <= 0:
            raise ValueError("all FFTformer integer hyperparameters must be positive")
        if not np.isfinite(float(self.ffn_expansion_factor)) or float(self.ffn_expansion_factor) <= 0.0:
            raise ValueError("ffn_expansion_factor must be a positive finite value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass(frozen=True)
class PlaNetModelConfig:
    model_family: str = "planet"
    input_channels: int = 1
    output_channels: int = 1
    base_channels: int = 32
    se_reduction: int = 8
    edge_loss_weight: float = 10.0

    def validate(self) -> None:
        if infer_model_family(self) != "planet":
            raise ValueError("PlaNetModelConfig.model_family must resolve to 'planet'")
        if self.input_channels <= 0 or self.output_channels <= 0:
            raise ValueError("PlaNet input/output channels must be positive")
        if self.base_channels <= 0:
            raise ValueError("PlaNet base_channels must be positive")
        if self.se_reduction <= 0:
            raise ValueError("PlaNet se_reduction must be positive")
        if not np.isfinite(self.edge_loss_weight) or self.edge_loss_weight < 0.0:
            raise ValueError("PlaNet edge_loss_weight must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class RDBMModelConfig:
    model_family: str = "rdbm"
    input_channels: int = 1
    output_channels: int = 1
    dim: int = 64
    timesteps: int = 100
    sampling_timesteps: int = 10

    def validate(self) -> None:
        if infer_model_family(self) != "rdbm":
            raise ValueError("RDBMModelConfig.model_family must resolve to 'rdbm'")
        if self.input_channels <= 0 or self.output_channels <= 0:
            raise ValueError("RDBM input/output channels must be positive")
        if self.input_channels != self.output_channels:
            raise ValueError("RDBM requires matching input/output channels")
        if self.dim <= 0:
            raise ValueError("RDBM dim must be positive")
        if self.timesteps <= 0 or self.sampling_timesteps <= 0:
            raise ValueError("RDBM timestep counts must be positive")
        if self.sampling_timesteps > self.timesteps:
            raise ValueError("RDBM sampling_timesteps cannot exceed timesteps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def infer_model_family(config: Any) -> str:
    if config is None:
        return "scgn"
    if isinstance(config, dict):
        value = config.get("model_family")
    else:
        value = getattr(config, "model_family", None)
    text = "" if value is None else str(value).strip().lower()
    return text or "scgn"


def materialize_model_config(config: Any) -> Any:
    family = infer_model_family(config)
    if isinstance(config, SCGNModelConfig):
        config.validate()
        return config
    if isinstance(config, ConvIRModelConfig):
        config.validate()
        return config
    if isinstance(config, StarIRModelConfig):
        config.validate()
        return config
    if isinstance(config, RestormerModelConfig):
        config.validate()
        return config
    if isinstance(config, MambaIRv2ModelConfig):
        config.validate()
        return config
    if isinstance(config, NAFNetModelConfig):
        config.validate()
        return config
    if isinstance(config, FFTformerModelConfig):
        config.validate()
        return config
    if isinstance(config, PlaNetModelConfig):
        config.validate()
        return config
    if isinstance(config, RDBMModelConfig):
        config.validate()
        return config
    payload = {} if config is None else dict(config)
    if family == "scgn":
        payload["model_family"] = "scgn"
        materialized = SCGNModelConfig(**payload)
        materialized.validate()
        return materialized
    if family == "convir":
        payload["model_family"] = "convir"
        materialized = ConvIRModelConfig(**payload)
        materialized.validate()
        return materialized
    if family == "starir":
        payload["model_family"] = "starir"
        materialized = StarIRModelConfig(**payload)
        materialized.validate()
        return materialized
    if family == "restormer":
        payload["model_family"] = "restormer"
        materialized = RestormerModelConfig(**payload)
        materialized.validate()
        return materialized
    if family == "mambairv2":
        payload["model_family"] = "mambairv2"
        materialized = MambaIRv2ModelConfig(**payload)
        materialized.validate()
        return materialized
    if family == "nafnet":
        payload["model_family"] = "nafnet"
        materialized = NAFNetModelConfig(**payload)
        materialized.validate()
        return materialized
    if family == "fftformer":
        payload["model_family"] = "fftformer"
        materialized = FFTformerModelConfig(**payload)
        materialized.validate()
        return materialized
    if family == "planet":
        payload["model_family"] = "planet"
        materialized = PlaNetModelConfig(**payload)
        materialized.validate()
        return materialized
    if family == "rdbm":
        payload["model_family"] = "rdbm"
        materialized = RDBMModelConfig(**payload)
        materialized.validate()
        return materialized
    raise ValueError(f"Unsupported model_family: {family}")

def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("restoration.py requires a working torch installation")


def _conv3x3(in_channels: int, out_channels: int, *, bias: bool = False) -> Any:
    _require_torch()
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        bias=bias,
        padding_mode="reflect",
    )


def _plain_conv3x3(in_channels: int, out_channels: int, *, bias: bool = False) -> Any:
    _require_torch()
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        bias=bias,
    )


class ConvBnRelu(ModuleBase):
    def __init__(self, channels: int, *, kernel_size: int = 3) -> None:
        _require_torch()
        super().__init__()
        if int(kernel_size) != 3:
            raise ValueError("ConvBnRelu currently expects kernel_size=3")
        self.conv = _conv3x3(channels, channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SpatialDeviationGuidedWeighting(ModuleBase):
    def __init__(self, channels: int, *, local_window: int = 3) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if local_window <= 0 or local_window % 2 == 0:
            raise ValueError("local_window must be a positive odd integer")
        self.local_window = int(local_window)
        self.conv = _conv3x3(channels, channels, bias=False)
        self.weight_proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def _local_std(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.local_window // 2
        if padding > 0:
            x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        mean = F.avg_pool2d(x, kernel_size=self.local_window, stride=1)
        mean_sq = F.avg_pool2d(x * x, kernel_size=self.local_window, stride=1)
        var = (mean_sq - mean * mean).clamp_min(0.0)
        return torch.sqrt(var + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deviation = self._local_std(x)
        weight = torch.sigmoid(self.weight_proj(deviation))
        guided = self.conv(x) * weight
        return self.relu(self.bn(guided))


class FrequencyBandGuidedWeighting(ModuleBase):
    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 8,
        pad_size: int = 16,
        init_scale: float = 0.1,
    ) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if pad_size < 0:
            raise ValueError("pad_size must be non-negative")
        if not np.isfinite(float(init_scale)):
            raise ValueError("init_scale must be finite")
        feature_channels = channels * 2
        squeeze_channels = max(feature_channels // int(reduction), 4)
        self.pad_size = int(pad_size)
        self.decouple = nn.Conv2d(feature_channels + 2, feature_channels, 1, bias=False)
        self.attn_down = nn.Conv2d(feature_channels, squeeze_channels, 1, bias=True)
        self.attn_up = nn.Conv2d(squeeze_channels, feature_channels, 1, bias=True)
        self.couple = nn.Conv2d(feature_channels, feature_channels, 1, bias=False)
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.output_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def _resolve_pad(self, height: int, width: int) -> tuple[int, int]:
        if self.pad_size <= 0:
            return 0, 0
        pad_h = min(self.pad_size, max(height - 1, 0))
        pad_w = min(self.pad_size, max(width - 1, 0))
        return int(pad_h), int(pad_w)

    @staticmethod
    def _band_position(
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        yy = y.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        xx = x.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return torch.cat([yy, xx], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h, pad_w = self._resolve_pad(height, width)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            if pad_h > 0 or pad_w > 0:
                x32 = F.pad(x32, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
            padded_height, padded_width = x32.shape[-2:]
            fft = torch.fft.rfft2(x32, norm="ortho")
            real = fft.real
            imag = fft.imag
            band = torch.cat([real, imag], dim=1)
            pos = self._band_position(
                batch_size=x32.shape[0],
                height=band.shape[-2],
                width=band.shape[-1],
                device=band.device,
                dtype=band.dtype,
            )
            decoupled = self.decouple(torch.cat([band, pos], dim=1))
            avg_pooled = F.adaptive_avg_pool2d(decoupled, 1)
            max_pooled = F.adaptive_max_pool2d(decoupled, 1)
            avg_weight = torch.sigmoid(self.attn_up(F.relu(self.attn_down(avg_pooled), inplace=True)))
            max_weight = torch.sigmoid(self.attn_up(F.relu(self.attn_down(max_pooled), inplace=True)))
            weight = avg_weight + max_weight
            coupled = self.couple(decoupled * weight)
            real_out, imag_out = torch.chunk(coupled, 2, dim=1)
            freq_out = torch.complex(real_out, imag_out)
            spatial = torch.fft.irfft2(freq_out, s=(padded_height, padded_width), norm="ortho")
            if pad_h > 0 or pad_w > 0:
                spatial = spatial[..., pad_h : pad_h + height, pad_w : pad_w + width]
        return self.relu(self.bn(spatial.to(dtype=x.dtype)))


class SpatialFrequencyEnhancementBlock(ModuleBase):
    def __init__(
        self,
        channels: int,
        *,
        branch_repeats: int = 1,
        reduction: int = 8,
        frequency_pad: int = 16,
        fbgw_init_scale: float = 0.1,
        use_sdgw: bool = True,
        use_fbgw: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0 or channels % 2 != 0:
            raise ValueError("channels must be a positive even integer")
        if int(branch_repeats) <= 0:
            raise ValueError("branch_repeats must be positive")
        branch_channels = channels // 2
        self.spatial_branch = (
            SpatialDeviationGuidedWeighting(branch_channels)
            if bool(use_sdgw)
            else ConvBnRelu(branch_channels, kernel_size=3)
        )
        self.extra_spatial_branches = nn.ModuleList(
            [
                (
                    SpatialDeviationGuidedWeighting(branch_channels)
                    if bool(use_sdgw)
                    else ConvBnRelu(branch_channels, kernel_size=3)
                )
                for _ in range(int(branch_repeats) - 1)
            ]
        )
        self.frequency_branch = (
            FrequencyBandGuidedWeighting(
                branch_channels,
                reduction=reduction,
                pad_size=frequency_pad,
                init_scale=fbgw_init_scale,
            )
            if bool(use_fbgw)
            else ConvBnRelu(branch_channels, kernel_size=3)
        )
        self.extra_frequency_branches = nn.ModuleList(
            [
                (
                    FrequencyBandGuidedWeighting(
                        branch_channels,
                        reduction=reduction,
                        pad_size=frequency_pad,
                        init_scale=fbgw_init_scale,
                    )
                    if bool(use_fbgw)
                    else ConvBnRelu(branch_channels, kernel_size=3)
                )
                for _ in range(int(branch_repeats) - 1)
            ]
        )
        self.fuse = _conv3x3(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_in, frequency_in = torch.chunk(x, 2, dim=1)
        spatial_out = self.spatial_branch(spatial_in)
        for branch in self.extra_spatial_branches:
            spatial_out = branch(spatial_out)
        frequency_out = self.frequency_branch(frequency_in)
        for branch in self.extra_frequency_branches:
            frequency_out = branch(frequency_out)
        fused = self.fuse(torch.cat([spatial_out, frequency_out], dim=1))
        return x + fused


class ReferenceWindowStd(ModuleBase):
    def __init__(self, channels: int, *, kernel_size: int = 3, eps: float = 1e-6) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.eps = float(eps)
        kernel = torch.full(
            (self.channels, 1, self.kernel_size, self.kernel_size),
            1.0 / float(self.kernel_size * self.kernel_size),
            dtype=torch.float32,
        )
        self.register_buffer("mean_kernel", kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AMP can overflow on bright 1024x1024 inputs when squaring fp16 values.
        # Keep the local-std path in fp32, then cast the result back.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            padded = F.pad(x32, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
            mean = F.conv2d(padded, self.mean_kernel, groups=self.channels)
            squared = x32 * x32
            squared_padded = F.pad(squared, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
            mean_squared = F.conv2d(squared_padded, self.mean_kernel, groups=self.channels)
            var = torch.clamp(mean_squared - mean * mean, min=self.eps)
            std = torch.sqrt(var)
        return std.to(dtype=x.dtype)


class ReferenceChannelAttention(ModuleBase):
    def __init__(self, channels: int, *, reduction: int = 16) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        hidden = max(channels // int(reduction), 4)
        self.conv = _plain_conv3x3(channels, channels, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))


class ReferenceFourierUnit(ModuleBase):
    def __init__(self, channels: int, *, reduction: int = 16, normalization: str = "batch") -> None:
        _require_torch()
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.conv1 = nn.Conv2d(self.channels * 2 + 2, self.channels * 2, kernel_size=1, bias=False)
        if normalization == "batch":
            self.bn = nn.BatchNorm2d(self.channels * 2)
        elif normalization == "channel_layer":
            self.bn = SCGNChannelLayerNorm2d(self.channels * 2)
        else:
            raise ValueError("normalization must be 'batch' or 'channel_layer'")
        self.relu = nn.ReLU(inplace=True)
        self.attn = ReferenceChannelAttention(self.channels * 2, reduction=reduction)
        self.conv2 = nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft_dim = (-2, -1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            batch = int(x32.shape[0])
            ffted = torch.fft.rfftn(x32, dim=fft_dim, norm="ortho")
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view(batch, -1, ffted.shape[-2], ffted.shape[-1])
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0.0, 1.0, steps=height, device=ffted.device, dtype=ffted.dtype)
            coords_hor = torch.linspace(0.0, 1.0, steps=width, device=ffted.device, dtype=ffted.dtype)
            coords_vert = coords_vert.view(1, 1, height, 1).expand(batch, 1, height, width)
            coords_hor = coords_hor.view(1, 1, 1, width).expand(batch, 1, height, width)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
            ffted = self.relu(self.bn(self.conv1(ffted)))
            ffted = ffted * self.attn(ffted)
            ffted = self.conv2(ffted)
            ffted = ffted.view(batch, -1, 2, ffted.shape[-2], ffted.shape[-1]).permute(0, 1, 3, 4, 2).contiguous()
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])
            output = torch.fft.irfftn(ffted, s=x32.shape[-2:], dim=fft_dim, norm="ortho")
        return output.to(dtype=x.dtype)


class ReferenceSpectralTransform(ModuleBase):
    def __init__(self, channels: int, *, reduction: int = 16, normalization: str = "batch") -> None:
        _require_torch()
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.conv1 = _plain_conv3x3(channels, channels, bias=True)
        self.fourier = ReferenceFourierUnit(channels, reduction=reduction, normalization=normalization)
        self.conv2 = _plain_conv3x3(channels * 2, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.conv1(x)
        spectral = self.fourier(local)
        return self.conv2(torch.cat([x, spectral], dim=1))


class ReferenceSpatialBranch(ModuleBase):
    def __init__(self, channels: int, *, use_sdgw: bool = True) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.use_sdgw = bool(use_sdgw)
        self.feature = _plain_conv3x3(channels, channels, bias=True)
        if self.use_sdgw:
            self.window_std = ReferenceWindowStd(channels, kernel_size=3)
            self.weight_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature(x)
        if not self.use_sdgw:
            return feature
        deviation = self.window_std(x)
        weight = self.sigmoid(self.weight_proj(deviation))
        return feature * weight


class ReferenceFFC(ModuleBase):
    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 8,
        use_sdgw: bool = True,
        use_fbgw: bool = True,
        normalization: str = "batch",
    ) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0 or channels % 2 != 0:
            raise ValueError("channels must be a positive even integer")
        branch_channels = channels // 2
        self.local_branch = ReferenceSpatialBranch(branch_channels, use_sdgw=use_sdgw)
        self.global_branch = (
            ReferenceSpectralTransform(branch_channels, reduction=reduction, normalization=normalization)
            if bool(use_fbgw)
            else _plain_conv3x3(branch_channels, branch_channels, bias=True)
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x_l, x_g = x
        else:
            x_l, x_g = torch.chunk(x, 2, dim=1)
        return self.local_branch(x_l), self.global_branch(x_g)


class ReferenceFFCBnAct(ModuleBase):
    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 8,
        use_sdgw: bool = True,
        use_fbgw: bool = True,
        normalization: str = "batch",
    ) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0 or channels % 2 != 0:
            raise ValueError("channels must be a positive even integer")
        branch_channels = channels // 2
        self.ffc = ReferenceFFC(
            channels,
            reduction=reduction,
            use_sdgw=use_sdgw,
            use_fbgw=use_fbgw,
            normalization=normalization,
        )
        if normalization == "batch":
            self.bn_l = nn.BatchNorm2d(branch_channels)
            self.bn_g = nn.BatchNorm2d(branch_channels)
        elif normalization == "channel_layer":
            self.bn_l = SCGNChannelLayerNorm2d(branch_channels)
            self.bn_g = SCGNChannelLayerNorm2d(branch_channels)
        else:
            raise ValueError("normalization must be 'batch' or 'channel_layer'")
        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = self.ffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))


class ReferenceFFCResnetBlock(ModuleBase):
    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 8,
        use_sdgw: bool = True,
        use_fbgw: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0 or channels % 2 != 0:
            raise ValueError("channels must be a positive even integer")
        self.stage1 = ReferenceFFCBnAct(
            channels,
            reduction=reduction,
            use_sdgw=use_sdgw,
            use_fbgw=use_fbgw,
        )
        self.stage2 = ReferenceFFCBnAct(
            channels,
            reduction=reduction,
            use_sdgw=use_sdgw,
            use_fbgw=use_fbgw,
        )
        self.fuse = _plain_conv3x3(channels, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l, x_g = torch.chunk(x, 2, dim=1)
        x_l, x_g = self.stage1((x_l, x_g))
        x_l, x_g = self.stage2((x_l, x_g))
        residual = self.fuse(torch.cat((x_l, x_g), dim=1))
        return x + residual


class PaperFFCResnetBlock(ModuleBase):
    """Faithful block layout from the authors' released SCGN implementation."""

    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 16,
        use_sdgw: bool = True,
        use_fbgw: bool = True,
        normalization: str = "batch",
    ) -> None:
        _require_torch()
        super().__init__()
        if channels <= 0 or channels % 2 != 0:
            raise ValueError("channels must be a positive even integer")
        self.stage1 = ReferenceFFCBnAct(
            channels,
            reduction=reduction,
            use_sdgw=use_sdgw,
            use_fbgw=use_fbgw,
            normalization=normalization,
        )
        self.stage2 = ReferenceFFCBnAct(
            channels,
            reduction=reduction,
            use_sdgw=use_sdgw,
            use_fbgw=use_fbgw,
            normalization=normalization,
        )
        self.fuse = _plain_conv3x3(channels, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l, x_g = torch.chunk(x, 2, dim=1)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.stage1((x_l, x_g))
        x_l, x_g = self.stage2((x_l, x_g))
        x_l = id_l + x_l
        x_g = id_g + x_g
        return self.fuse(torch.cat((x_l, x_g), dim=1))


class SCGNMidBandNAFRefinement(ModuleBase):
    """Route a local NAF correction using cross-stage structure and band residuals."""

    def __init__(
        self,
        channels: int,
        *,
        low_cut: float,
        mid_cut: float,
        num_blocks: int = 1,
        router_channels: int = 16,
    ) -> None:
        _require_torch()
        super().__init__()
        from rawsr.starir import StarIRLayerNorm, StarIRNAFRefinement

        self.low_cut = float(low_cut)
        self.mid_cut = float(mid_cut)
        self.midpoint_projection = nn.Conv2d(
            int(channels),
            int(channels),
            kernel_size=1,
            bias=False,
        )
        nn.init.dirac_(self.midpoint_projection.weight)
        self.midpoint_norm = StarIRLayerNorm(int(channels), "BiasFree")
        self.final_norm = StarIRLayerNorm(int(channels), "BiasFree")
        self.band_residual_norm = StarIRLayerNorm(int(channels), "BiasFree")
        self.local_expert = nn.Sequential(
            *[
                StarIRNAFRefinement(int(channels), layernorm_type="BiasFree")
                for _ in range(int(num_blocks))
            ]
        )
        self.router = nn.Sequential(
            nn.Conv2d(2, int(router_channels), kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(int(router_channels), 3, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def routing_weights(
        self,
        midpoint_features: torch.Tensor,
        pre_band_features: torch.Tensor,
        band_features: torch.Tensor,
    ) -> torch.Tensor:
        projected_midpoint = self.midpoint_projection(midpoint_features)
        if projected_midpoint.shape[-2:] != band_features.shape[-2:]:
            projected_midpoint = F.interpolate(
                projected_midpoint,
                size=band_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        structure_disagreement = (
            self.final_norm(band_features) - self.midpoint_norm(projected_midpoint)
        ).abs().mean(dim=1, keepdim=True)
        band_residual = self.band_residual_norm(
            band_features - pre_band_features
        ).abs().mean(dim=1, keepdim=True)
        return torch.softmax(
            self.router(torch.cat((structure_disagreement, band_residual), dim=1)),
            dim=1,
        )

    def _frequency_bands(self, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from rawsr.starir import StarIRBandResidualAdapter

        with torch.autocast(device_type=residual.device.type, enabled=False):
            spectrum = torch.fft.rfft2(residual.float(), norm="ortho")
            masks = StarIRBandResidualAdapter.radial_masks(
                residual.shape[-2],
                residual.shape[-1],
                device=residual.device,
                low_cut=self.low_cut,
                mid_cut=self.mid_cut,
            )
            bands = tuple(
                torch.fft.irfft2(
                    spectrum * mask,
                    s=residual.shape[-2:],
                    norm="ortho",
                ).to(dtype=residual.dtype)
                for mask in masks
            )
        return bands

    def forward(
        self,
        midpoint_features: torch.Tensor,
        pre_band_features: torch.Tensor,
        band_features: torch.Tensor,
    ) -> torch.Tensor:
        candidate_residual = self.local_expert(band_features) - band_features
        weights = self.routing_weights(midpoint_features, pre_band_features, band_features)
        routed_residual = sum(
            weights[:, index : index + 1] * band
            for index, band in enumerate(self._frequency_bands(candidate_residual))
        )
        return band_features + routed_residual


class SCGNLocalPatchFourierRefinement(ModuleBase):
    """Learn a shifted-window complex residual without PSF or OTF supervision."""

    def __init__(
        self,
        channels: int,
        *,
        patch_size: int = 64,
        expert_channels: int = 8,
        chunk_size: int = 16,
    ) -> None:
        _require_torch()
        super().__init__()
        if int(patch_size) <= 0 or int(patch_size) % 2 != 0:
            raise ValueError("patch_size must be a positive even integer")
        if int(expert_channels) <= 0:
            raise ValueError("expert_channels must be positive")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.patch_size = int(patch_size)
        self.expert_channels = int(expert_channels)
        self.chunk_size = int(chunk_size)
        self.input_projection = nn.Conv2d(
            int(channels),
            self.expert_channels,
            kernel_size=1,
            bias=False,
        )
        complex_channels = self.expert_channels * 2
        self.frequency_mixer = nn.Sequential(
            nn.Conv2d(complex_channels + 2, complex_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(complex_channels, complex_channels, kernel_size=1, bias=False),
        )
        self.output_projection = nn.Conv2d(
            self.expert_channels,
            int(channels),
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.output_projection.weight)

    def _partition(
        self,
        x: torch.Tensor,
        *,
        offset: int,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int]]:
        patch = self.patch_size
        batch, channels, height, width = x.shape
        top = int(offset)
        left = int(offset)
        bottom = (-(height + top)) % patch
        right = (-(width + left)) % patch
        pad_mode = "reflect" if min(height, width) > max(top, left, bottom, right) else "replicate"
        padded = F.pad(x, (left, right, top, bottom), mode=pad_mode)
        padded_height, padded_width = padded.shape[-2:]
        patches = (
            padded.reshape(
                batch,
                channels,
                padded_height // patch,
                patch,
                padded_width // patch,
                patch,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, channels, patch, patch)
        )
        layout = (batch, channels, height, width, padded_height, padded_width)
        return patches, layout

    def _merge(
        self,
        patches: torch.Tensor,
        layout: tuple[int, int, int, int, int, int],
        *,
        offset: int,
    ) -> torch.Tensor:
        batch, channels, height, width, padded_height, padded_width = layout
        patch = self.patch_size
        merged = (
            patches.reshape(
                batch,
                padded_height // patch,
                padded_width // patch,
                channels,
                patch,
                patch,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, padded_height, padded_width)
        )
        top = int(offset)
        left = int(offset)
        return merged[..., top : top + height, left : left + width]

    def _frequency_coordinates(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        patch = self.patch_size
        vertical = torch.linspace(-1.0, 1.0, patch, device=device, dtype=dtype)
        horizontal = torch.linspace(0.0, 1.0, patch // 2 + 1, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
        return torch.stack((grid_y, grid_x), dim=0).unsqueeze(0)

    def _process_patch_chunk(self, patches: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=patches.device.type, enabled=False):
            spectrum = torch.fft.rfft2(patches.float(), norm="ortho")
            real_imag = torch.cat((spectrum.real, spectrum.imag), dim=1)
            coordinates = self._frequency_coordinates(
                device=patches.device,
                dtype=real_imag.dtype,
            ).expand(real_imag.shape[0], -1, -1, -1)
            mixed = self.frequency_mixer(torch.cat((real_imag, coordinates), dim=1))
            real, imag = mixed.chunk(2, dim=1)
            transformed = torch.fft.irfft2(
                torch.complex(real, imag),
                s=(self.patch_size, self.patch_size),
                norm="ortho",
            )
        return transformed.to(dtype=patches.dtype)

    def _transform_tiling(self, x: torch.Tensor, *, offset: int) -> torch.Tensor:
        patches, layout = self._partition(x, offset=offset)
        transformed = [
            self._process_patch_chunk(patches[start : start + self.chunk_size])
            for start in range(0, patches.shape[0], self.chunk_size)
        ]
        return self._merge(torch.cat(transformed, dim=0), layout, offset=offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        primary = self._transform_tiling(projected, offset=0)
        shifted = self._transform_tiling(projected, offset=self.patch_size // 2)
        correction = self.output_projection((primary + shifted) * 0.5)
        return x + correction


class SCGNRestorationNet(ModuleBase):
    def __init__(self, config: SCGNModelConfig | None = None, **override_kwargs: Any) -> None:
        _require_torch()
        super().__init__()
        if config is None:
            config = SCGNModelConfig(**override_kwargs)
        elif override_kwargs:
            config = SCGNModelConfig(**{**config.to_dict(), **override_kwargs})
        config.validate()
        self.config = config
        self.band_adapter: ModuleBase = nn.Identity()
        self.use_source_conditioning = bool(config.use_source_conditioning)
        self.source_residual_scale = float(config.source_residual_scale)
        self.source_residual_target = str(config.source_residual_target).strip().lower()
        self.supports_source_conditioning = self.use_source_conditioning
        self.source_residual_head = None
        self.use_midpoint_aux = bool(config.use_midpoint_aux)
        self.supports_midpoint_auxiliary = self.use_midpoint_aux
        self.midpoint_aux_head = None
        self.use_naf_refinement = bool(config.use_naf_refinement)
        self.naf_refinement: ModuleBase = nn.Identity()
        self.use_midband_naf_refinement = bool(config.use_midband_naf_refinement)
        self.midband_naf_refinement: ModuleBase = nn.Identity()
        self.use_local_patch_fourier_refinement = bool(
            config.use_local_patch_fourier_refinement
        )
        self.local_patch_fourier_refinement: ModuleBase = nn.Identity()
        if config.variant == "reference":
            self.head_conv = _plain_conv3x3(config.input_channels, config.hidden_channels, bias=True)
            self.body = nn.Sequential(
                *[
                    ReferenceFFCResnetBlock(
                        config.hidden_channels,
                        reduction=config.reduction,
                        use_sdgw=config.use_sdgw,
                        use_fbgw=config.use_fbgw,
                    )
                    for _ in range(config.num_blocks)
                ]
            )
            self.tail_conv = _plain_conv3x3(config.hidden_channels, config.output_channels, bias=True)
        elif config.variant in {"paper", "paper_residual", "paper_residual_ln"}:
            normalization = "channel_layer" if config.variant == "paper_residual_ln" else "batch"
            self.head_conv = _plain_conv3x3(config.input_channels, config.hidden_channels, bias=True)
            self.body = nn.ModuleList(
                [
                    PaperFFCResnetBlock(
                        config.hidden_channels,
                        reduction=config.reduction,
                        use_sdgw=config.use_sdgw,
                        use_fbgw=config.use_fbgw,
                        normalization=normalization,
                    )
                    for _ in range(config.num_blocks)
                ]
            )
            self.use_activation_checkpoint = (
                config.variant == "paper_residual_ln"
                and activation_checkpoint is not None
            )
            if bool(config.use_band_adapter):
                # This dependency is only required by the optional SAFIR adapter.
                from rawsr.starir import StarIRBandResidualAdapter

                self.band_adapter = StarIRBandResidualAdapter(
                    config.hidden_channels,
                    low_cut=float(config.band_low_cut),
                    mid_cut=float(config.band_mid_cut),
                )
            self.tail_conv = _plain_conv3x3(config.hidden_channels, config.output_channels, bias=True)
            if config.variant in {"paper_residual", "paper_residual_ln"}:
                nn.init.zeros_(self.tail_conv.weight)
                nn.init.zeros_(self.tail_conv.bias)
            if self.use_source_conditioning:
                self.source_residual_head = nn.Conv2d(
                    config.hidden_channels,
                    int(config.output_channels) * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                nn.init.zeros_(self.source_residual_head.weight)
                nn.init.zeros_(self.source_residual_head.bias)
            if self.use_midpoint_aux:
                # Construct this after all primary-path modules so paired seeds
                # initialize every shared parameter identically to the control.
                self.midpoint_aux_head = nn.Conv2d(
                    config.hidden_channels,
                    config.output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                nn.init.zeros_(self.midpoint_aux_head.weight)
                nn.init.zeros_(self.midpoint_aux_head.bias)
            if self.use_naf_refinement:
                # Keep all shared primary modules ahead of this optional branch so
                # paired seeds produce identical SAFIR+A2BAND initial states.
                from rawsr.starir import StarIRNAFRefinement

                self.naf_refinement = nn.Sequential(
                    *[
                        StarIRNAFRefinement(
                            config.hidden_channels,
                            layernorm_type="BiasFree",
                        )
                        for _ in range(int(config.naf_refinement_blocks))
                    ]
                )
            if self.use_midband_naf_refinement:
                self.midband_naf_refinement = SCGNMidBandNAFRefinement(
                    config.hidden_channels,
                    low_cut=float(config.band_low_cut),
                    mid_cut=float(config.band_mid_cut),
                    num_blocks=int(config.midband_naf_refinement_blocks),
                )
            if self.use_local_patch_fourier_refinement:
                self.local_patch_fourier_refinement = SCGNLocalPatchFourierRefinement(
                    config.hidden_channels,
                    patch_size=int(config.local_patch_fourier_size),
                    expert_channels=int(config.local_patch_fourier_expert_channels),
                    chunk_size=int(config.local_patch_fourier_chunk_size),
                )
        else:
            self.entry = _conv3x3(config.input_channels, config.hidden_channels, bias=False)
            self.blocks = nn.ModuleList(
                [
                    SpatialFrequencyEnhancementBlock(
                        config.hidden_channels,
                        branch_repeats=config.branch_repeats,
                        reduction=config.reduction,
                        frequency_pad=config.frequency_pad,
                        fbgw_init_scale=config.fbgw_init_scale,
                        use_sdgw=config.use_sdgw,
                        use_fbgw=config.use_fbgw,
                    )
                    for _ in range(config.num_blocks)
                ]
            )
            self.exit = _conv3x3(config.hidden_channels, config.output_channels, bias=False)

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

    def _apply_source_residual(
        self,
        *,
        base_output: torch.Tensor,
        features: torch.Tensor,
        source_kind: list[str] | tuple[str, ...] | torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_source_conditioning or self.source_residual_head is None:
            return base_output
        source_indices = self._source_indices(source_kind, base_output.shape[0], base_output.device)
        source_residual = self.source_residual_head(features)
        source_residual = source_residual.view(
            base_output.shape[0],
            2,
            -1,
            base_output.shape[-2],
            base_output.shape[-1],
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
        return base_output + self.source_residual_scale * selected_residual

    def forward(
        self,
        x: torch.Tensor,
        source_kind: list[str] | tuple[str, ...] | torch.Tensor | None = None,
        return_midpoint_aux: bool = False,
    ) -> Any:
        if return_midpoint_aux and not self.supports_midpoint_auxiliary:
            raise ValueError("midpoint auxiliary output was requested but is not configured")
        midpoint_features = None
        if self.config.variant == "reference":
            feat = self.head_conv(x)
            shallow = feat
            feat = self.body(feat)
            if self.config.global_feature_residual:
                feat = feat + shallow
            out = self.tail_conv(feat)
        elif self.config.variant in {"paper", "paper_residual", "paper_residual_ln"}:
            # Match the released SCGN implementation's body layout and single
            # global shallow skip for all paper-family variants.
            feat = self.head_conv(x)
            shallow = feat
            for block_idx, block in enumerate(self.body, start=1):
                if getattr(self, "use_activation_checkpoint", False) and self.training:
                    feat = activation_checkpoint(block, feat, use_reentrant=False)
                else:
                    feat = block(feat)
                if self.use_midpoint_aux and block_idx == int(self.config.midpoint_aux_block):
                    midpoint_features = feat
            feat = feat + shallow
            pre_band_features = feat
            feat = self.band_adapter(feat)
            feat = self.naf_refinement(feat)
            if self.use_midband_naf_refinement:
                if midpoint_features is None:
                    raise RuntimeError("midband NAF refinement requires midpoint features")
                if getattr(self, "use_activation_checkpoint", False) and self.training:
                    feat = activation_checkpoint(
                        self.midband_naf_refinement,
                        midpoint_features,
                        pre_band_features,
                        feat,
                        use_reentrant=False,
                    )
                else:
                    feat = self.midband_naf_refinement(
                        midpoint_features,
                        pre_band_features,
                        feat,
                    )
            if self.use_local_patch_fourier_refinement:
                if getattr(self, "use_activation_checkpoint", False) and self.training:
                    feat = activation_checkpoint(
                        self.local_patch_fourier_refinement,
                        feat,
                        use_reentrant=False,
                    )
                else:
                    feat = self.local_patch_fourier_refinement(feat)
            out = self.tail_conv(feat)
            if self.config.variant in {"paper_residual", "paper_residual_ln"}:
                out = x + out
            out = self._apply_source_residual(
                base_output=out,
                features=feat,
                source_kind=source_kind,
            )
        else:
            feat = self.entry(x)
            shallow = feat
            for block in self.blocks:
                feat = block(feat)
            if self.config.global_feature_residual:
                feat = feat + shallow
            out = self.exit(feat)
        if (
            self.config.variant not in {"paper", "paper_residual", "paper_residual_ln"}
            and self.config.global_output_residual
            and self.config.input_channels == self.config.output_channels
        ):
            out = out + x
        if return_midpoint_aux:
            if midpoint_features is None or self.midpoint_aux_head is None:
                raise RuntimeError("configured midpoint auxiliary features were not produced")
            midpoint_prediction = x + self.midpoint_aux_head(midpoint_features)
            return out, midpoint_prediction
        return out


def build_restoration_model(config: Any, **override_kwargs: Any) -> ModuleBase:
    materialized = materialize_model_config(config)
    if override_kwargs:
        payload = materialized.to_dict()
        payload.update(override_kwargs)
        materialized = materialize_model_config(payload)
    family = infer_model_family(materialized)
    if family == "scgn":
        return SCGNRestorationNet(materialized)
    if family == "convir":
        from baseline.ConvIR.code.convir import ConvIRRestorationNet

        return ConvIRRestorationNet(
            input_channels=int(materialized.input_channels),
            output_channels=int(materialized.output_channels),
            base_channel=int(materialized.base_channel),
            num_res=int(materialized.num_res),
        )
    if family == "starir":
        from baseline.StarIR.code.starir import StarIRRestorationNet

        return StarIRRestorationNet(
            input_channels=int(materialized.input_channels),
            output_channels=int(materialized.output_channels),
            dim=int(materialized.dim),
            num_blocks=(
                int(materialized.num_blocks_l1),
                int(materialized.num_blocks_l2),
                int(materialized.num_blocks_l3),
            ),
            num_refinement_blocks=int(materialized.num_refinement_blocks),
            ffn_expansion_factor=float(materialized.ffn_expansion_factor),
            bias=bool(materialized.bias),
            use_band_adapter=bool(materialized.use_band_adapter),
            band_low_cut=float(materialized.band_low_cut),
            band_mid_cut=float(materialized.band_mid_cut),
            use_ofr=bool(materialized.use_ofr),
            ofr_prompt_channels=int(materialized.ofr_prompt_channels),
            ofr_expert_channels=int(materialized.ofr_expert_channels),
            ofr_window_size=int(materialized.ofr_window_size),
            ofr_temperature=float(materialized.ofr_temperature),
            ofr_residual_scale=float(materialized.ofr_residual_scale),
            use_source_conditioning=bool(materialized.use_source_conditioning),
            source_residual_scale=float(materialized.source_residual_scale),
            source_residual_target=str(materialized.source_residual_target),
            use_enc3_aux=bool(materialized.use_enc3_aux),
            use_enc3_blur_aux=bool(materialized.use_enc3_blur_aux),
            use_naf_refinement=bool(materialized.use_naf_refinement),
            naf_refinement_blocks=int(materialized.naf_refinement_blocks),
        )
    if family == "restormer":
        from baseline.Restormer.code.restormer import RestormerNet

        return RestormerNet(
            inp_channels=int(materialized.input_channels),
            out_channels=int(materialized.output_channels),
            dim=int(materialized.dim),
            num_blocks=(
                int(materialized.num_blocks_l1),
                int(materialized.num_blocks_l2),
                int(materialized.num_blocks_l3),
                int(materialized.num_blocks_latent),
            ),
            num_refinement_blocks=int(materialized.num_refinement_blocks),
            heads=(
                int(materialized.num_heads_l1),
                int(materialized.num_heads_l2),
                int(materialized.num_heads_l3),
                int(materialized.num_heads_latent),
            ),
            ffn_expansion_factor=float(materialized.ffn_expansion_factor),
            bias=bool(materialized.bias),
        )
    if family == "mambairv2":
        from baseline.MambaIRv2.code.mambairv2 import MambaIRv2

        return MambaIRv2(
            img_size=int(materialized.img_size),
            patch_size=int(materialized.patch_size),
            in_chans=int(materialized.input_channels),
            embed_dim=int(materialized.embed_dim),
            d_state=int(materialized.d_state),
            depths=(
                int(materialized.depth_l1),
                int(materialized.depth_l2),
                int(materialized.depth_l3),
                int(materialized.depth_l4),
            ),
            num_heads=(
                int(materialized.num_heads_l1),
                int(materialized.num_heads_l2),
                int(materialized.num_heads_l3),
                int(materialized.num_heads_l4),
            ),
            window_size=int(materialized.window_size),
            inner_rank=int(materialized.inner_rank),
            num_tokens=int(materialized.num_tokens),
            convffn_kernel_size=int(materialized.convffn_kernel_size),
            mlp_ratio=float(materialized.mlp_ratio),
            qkv_bias=bool(materialized.qkv_bias),
            patch_norm=bool(materialized.patch_norm),
            use_checkpoint=bool(materialized.use_checkpoint),
            upscale=1,
            img_range=1.0,
            upsampler="",
            resi_connection=str(materialized.resi_connection),
        )
    if family == "planet":
        from baseline.PlaNet.code.planet import build_planet_model

        return build_planet_model(materialized)
    if family == "rdbm":
        from baseline.RDBM.code.rdbm_adapter import build_rdbm_model

        return build_rdbm_model(materialized)

    if family == "nafnet":
        from baseline.NAFNet.code.nafnet import NAFNetRestorationNet

        return NAFNetRestorationNet(
            input_channels=int(materialized.input_channels),
            output_channels=int(materialized.output_channels),
            width=int(materialized.width),
            encoder_blocks=(
                int(materialized.encoder_blocks_l1),
                int(materialized.encoder_blocks_l2),
                int(materialized.encoder_blocks_l3),
            ),
            middle_blocks=int(materialized.middle_blocks),
            decoder_blocks=(
                int(materialized.decoder_blocks_l1),
                int(materialized.decoder_blocks_l2),
                int(materialized.decoder_blocks_l3),
            ),
            dw_expand=float(materialized.dw_expand),
            ffn_expand=float(materialized.ffn_expand),
        )
    if family == "fftformer":
        from baseline.FFTformer.code.fftformer import FFTformerRestorationNet

        return FFTformerRestorationNet(
            input_channels=int(materialized.input_channels),
            output_channels=int(materialized.output_channels),
            dim=int(materialized.dim),
            num_blocks=(
                int(materialized.num_blocks_l1),
                int(materialized.num_blocks_l2),
                int(materialized.num_blocks_l3),
                int(materialized.num_blocks_latent),
            ),
            num_refinement_blocks=int(materialized.num_refinement_blocks),
            ffn_expansion_factor=float(materialized.ffn_expansion_factor),
            bias=bool(materialized.bias),
        )
    raise ValueError(f"Unsupported model_family: {family}")
def model_output_list(pred: Any) -> list[torch.Tensor]:
    _require_torch()
    if isinstance(pred, torch.Tensor):
        return [pred]
    if isinstance(pred, (list, tuple)) and pred:
        outputs = list(pred)
        if not all(isinstance(item, torch.Tensor) for item in outputs):
            raise TypeError("all model outputs must be torch.Tensor instances")
        return outputs
    raise TypeError("model output must be a torch.Tensor or a non-empty sequence of torch.Tensor")


def primary_prediction(pred: Any) -> torch.Tensor:
    return model_output_list(pred)[-1]


def select_model_output_batch(pred: Any, indices: torch.Tensor) -> Any:
    outputs = [item.index_select(0, indices) for item in model_output_list(pred)]
    if isinstance(pred, torch.Tensor):
        return outputs[0]
    if isinstance(pred, tuple):
        return tuple(outputs)
    return outputs


class MaterializedRestorationDataset(TorchDataset):
    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
        real_root: str | Path | None = None,
        png_root: str | Path | None = None,
        real_include_paths_file: str | Path | None = None,
        real_exclude_paths_file: str | Path | None = None,
        png_include_paths_file: str | Path | None = None,
        png_exclude_paths_file: str | Path | None = None,
        real_source_metadata_file: str | Path | None = None,
        png_source_metadata_file: str | Path | None = None,
        source_mode: str = "both",
        mix_mode: str = "balanced",
        seed: int = 0,
        crop_size: int | None = None,
        random_crop: bool = True,
        intensity_scale: float = 1.0,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
        include_blurred: bool = False,
        blurred_root: str | Path | None = None,
        blurred_real_root: str | Path | None = None,
        blurred_png_root: str | Path | None = None,
    ) -> None:
        _require_torch()
        if intensity_scale <= 0.0:
            raise ValueError("intensity_scale must be positive")
        self.base = FITSMaterializedTrainDataset(
            root_dir=root_dir,
            real_root=real_root,
            png_root=png_root,
            real_include_paths_file=real_include_paths_file,
            real_exclude_paths_file=real_exclude_paths_file,
            png_include_paths_file=png_include_paths_file,
            png_exclude_paths_file=png_exclude_paths_file,
            real_source_metadata_file=real_source_metadata_file,
            png_source_metadata_file=png_source_metadata_file,
            source_mode=source_mode,
            mix_mode=mix_mode,
            seed=seed,
            as_tensor=False,
        )
        self.crop_size = None if crop_size is None or int(crop_size) <= 0 else int(crop_size)
        self.random_crop = bool(random_crop)
        self.intensity_scale = float(intensity_scale)
        self.clamp_min = None if clamp_min is None else float(clamp_min)
        self.clamp_max = None if clamp_max is None else float(clamp_max)
        self.include_blurred = bool(include_blurred)
        self.blurred_root = None if blurred_root is None else Path(blurred_root)
        self.blurred_real_root = None if blurred_real_root is None else Path(blurred_real_root)
        self.blurred_png_root = None if blurred_png_root is None else Path(blurred_png_root)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def source_counts(self) -> dict[str, int]:
        return self.base.source_counts

    @property
    def plan_counts(self) -> dict[str, int]:
        return self.base.plan_counts

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.clamp_min is not None or self.clamp_max is not None:
            min_value = self.clamp_min if self.clamp_min is not None else float("-inf")
            max_value = self.clamp_max if self.clamp_max is not None else float("inf")
            tensor = tensor.clamp(min=min_value, max=max_value)
        if self.intensity_scale != 1.0:
            tensor = tensor / self.intensity_scale
        return tensor

    def _crop_pair(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        blurred_tensor: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.crop_size is None:
            return input_tensor, target_tensor, blurred_tensor
        _, height, width = input_tensor.shape
        if self.crop_size > min(height, width):
            raise ValueError(f"crop_size={self.crop_size} is larger than sample size {(height, width)}")
        if self.crop_size == height and self.crop_size == width:
            return input_tensor, target_tensor, blurred_tensor
        if self.random_crop:
            y0 = int(torch.randint(0, height - self.crop_size + 1, (1,)).item())
            x0 = int(torch.randint(0, width - self.crop_size + 1, (1,)).item())
        else:
            y0 = (height - self.crop_size) // 2
            x0 = (width - self.crop_size) // 2
        slc_y = slice(y0, y0 + self.crop_size)
        slc_x = slice(x0, x0 + self.crop_size)
        input_crop = input_tensor[:, slc_y, slc_x]
        target_crop = target_tensor[:, slc_y, slc_x]
        blurred_crop = None if blurred_tensor is None else blurred_tensor[:, slc_y, slc_x]
        return input_crop, target_crop, blurred_crop

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[int(idx)]
        input_tensor = torch.from_numpy(array_from_sample(sample["input"]).astype(np.float32, copy=False)).clone()
        target_tensor = torch.from_numpy(array_from_sample(sample["target"]).astype(np.float32, copy=False)).clone()
        blurred_tensor = None
        resolved_blurred_path = str(sample.get("blurred_path") or "")
        if self.include_blurred:
            if self.blurred_real_root is not None or self.blurred_png_root is not None or self.blurred_root is not None:
                source_kind = str(sample.get("source_kind", "")).lower()
                target_root = (
                    self.blurred_real_root if source_kind == "real" else self.blurred_png_root
                ) or self.blurred_root
                if target_root is None:
                    raise RuntimeError("No midpoint blurred target root is configured")
                filename = Path(str(sample["input_path"])).name
                candidates = (target_root / "blurred" / filename, target_root / filename)
                resolved_path = next((path for path in candidates if path.is_file()), None)
                if resolved_path is None:
                    raise RuntimeError(
                        "Configured midpoint blurred target is missing: "
                        f"source_kind={source_kind} input={sample.get('input_path', '<unknown>')} "
                        f"root={target_root}"
                    )
                blurred_array = load_fits_array(resolved_path, memmap=True).astype(np.float32, copy=False)
                if blurred_array.ndim == 2:
                    blurred_array = blurred_array[None, ...]
                if blurred_array.ndim != 3:
                    raise ValueError(
                        "Configured midpoint blurred target must have shape [C,H,W] or [H,W]: "
                        f"path={resolved_path} shape={blurred_array.shape}"
                    )
                blurred_tensor = torch.from_numpy(blurred_array).clone()
                resolved_blurred_path = str(resolved_path)
            else:
                if "blurred" not in sample:
                    raise RuntimeError(
                        "include_blurred=True but the materialized sample has no blurred target: "
                        f"{sample.get('input_path', '<unknown>')}"
                    )
                blurred_tensor = torch.from_numpy(
                    array_from_sample(sample["blurred"]).astype(np.float32, copy=False)
                ).clone()
        input_tensor, target_tensor, blurred_tensor = self._crop_pair(input_tensor, target_tensor, blurred_tensor)
        payload: dict[str, Any] = {
            "input": self._normalize(input_tensor),
            "target": self._normalize(target_tensor),
            "has_blurred": bool(blurred_tensor is not None),
            "source_kind": str(sample["source_kind"]),
            "name": Path(str(sample["input_path"])).name,
            "input_path": str(sample["input_path"]),
            "target_path": str(sample["target_path"]),
        }
        # Keep metadata keys present in every item so the default DataLoader
        # collate function can mix real and PNG samples safely.
        for provenance_key in ("source_path", "source_name"):
            payload[provenance_key] = str(sample.get(provenance_key) or "")
        payload["blurred_path"] = resolved_blurred_path
        if blurred_tensor is not None:
            payload["blurred"] = self._normalize(blurred_tensor)
        return payload


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * int(weight)
        self.count += int(weight)

    @property
    def avg(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total / self.count


def gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require_torch()
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def fft_magnitude_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require_torch()
    pred_fft = torch.fft.rfft2(pred.float(), norm="ortho").abs()
    target_fft = torch.fft.rfft2(target.float(), norm="ortho").abs()
    return F.l1_loss(pred_fft, target_fft)


def resize_target_to_prediction(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    _require_torch()
    if target.shape[-2:] == pred.shape[-2:]:
        return target
    return F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)


def build_restoration_loss(
    pred: Any,
    target: torch.Tensor,
    *,
    gradient_weight: float = 0.0,
    fft_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    _require_torch()
    outputs = model_output_list(pred)
    pixel_l1 = target.new_zeros(())
    gradient_component = target.new_zeros(())
    fft_component = target.new_zeros(())
    for output in outputs:
        matched_target = resize_target_to_prediction(target, output)
        pixel_l1 = pixel_l1 + F.l1_loss(output, matched_target)
        if gradient_weight > 0.0:
            gradient_component = gradient_component + gradient_l1_loss(output, matched_target)
        if fft_weight > 0.0:
            fft_component = fft_component + fft_magnitude_l1_loss(output, matched_target)
    total = pixel_l1 + float(gradient_weight) * gradient_component + float(fft_weight) * fft_component
    return {
        "loss": total,
        "pixel_l1": pixel_l1,
        "gradient_l1": gradient_component,
        "fft_l1": fft_component,
    }


def detach_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    _require_torch()
    return {key: float(value.detach().cpu().item()) for key, value in metrics.items()}


def per_source_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_kinds: list[str],
) -> dict[str, AverageMeter]:
    _require_torch()
    if pred.shape[0] != len(source_kinds):
        raise ValueError("source_kinds length must match batch size")
    sample_values = torch.abs(pred - target).mean(dim=(1, 2, 3)).detach().cpu().tolist()
    meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
    for source_kind, value in zip(source_kinds, sample_values):
        meters[str(source_kind)].update(float(value), 1)
    return dict(meters)
