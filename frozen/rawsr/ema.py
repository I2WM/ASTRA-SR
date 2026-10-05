from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import torch
from torch import nn


class ParameterEMA:
    """Exponential moving average over the model's trainable named parameters."""

    def __init__(self, model: nn.Module, *, decay: float) -> None:
        resolved_decay = float(decay)
        if not 0.0 <= resolved_decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = resolved_decay
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA requires at least one trainable named parameter")
        self._swap_active = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        if set(parameters).intersection(self.shadow) != set(self.shadow):
            missing = sorted(set(self.shadow).difference(parameters))
            raise ValueError(f"EMA model is missing tracked parameters: {missing}")
        for name, shadow in self.shadow.items():
            value = parameters[name].detach().to(device=shadow.device, dtype=torch.float32)
            shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    @contextmanager
    @torch.no_grad()
    def swap_parameters(self, model: nn.Module) -> Iterator[None]:
        if self._swap_active:
            raise RuntimeError("EMA parameter swap is already active")
        parameters = dict(model.named_parameters())
        missing = sorted(set(self.shadow).difference(parameters))
        if missing:
            raise ValueError(f"EMA model is missing tracked parameters: {missing}")
        originals = {
            name: parameters[name].detach().clone()
            for name in self.shadow
        }
        self._swap_active = True
        try:
            for name, shadow in self.shadow.items():
                parameters[name].copy_(shadow.to(device=parameters[name].device, dtype=parameters[name].dtype))
            yield
        finally:
            for name, original in originals.items():
                parameters[name].copy_(original)
            self._swap_active = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": float(self.decay),
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any], model: nn.Module) -> None:
        decay = float(state["decay"])
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        incoming = dict(state["shadow"])
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(incoming) != set(parameters):
            missing = sorted(set(parameters).difference(incoming))
            unexpected = sorted(set(incoming).difference(parameters))
            raise ValueError(
                f"EMA state parameter mismatch: missing={missing}, unexpected={unexpected}"
            )
        loaded: dict[str, torch.Tensor] = {}
        for name, parameter in parameters.items():
            value = incoming[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"EMA state shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            loaded[name] = value.detach().to(device=parameter.device, dtype=torch.float32).clone()
        self.decay = decay
        self.shadow = loaded


__all__ = ["ParameterEMA"]
