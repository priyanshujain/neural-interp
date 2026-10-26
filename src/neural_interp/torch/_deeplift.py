from collections.abc import Callable

import numpy as np
import torch
from captum.attr import DeepLift
from torch import nn

from neural_interp.typing import FloatArray


class _Unit(nn.Module):
    def __init__(self, model: nn.Module, module: str) -> None:
        super().__init__()
        self.model = model
        self.index: tuple[int, ...] = ()
        self.output: torch.Tensor | None = None
        dict(model.named_modules())[module].register_forward_hook(self._grab)

    def _grab(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self.output = output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model(inputs)
        assert self.output is not None
        return self.output[(slice(None), *self.index)][:, None]


class UnitDeepLift:
    def __init__(
        self,
        model: nn.Module,
        module: str,
        reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.unit = _Unit(model, module)
        self.deeplift = DeepLift(self.unit)
        self.reduce = reduce

    def attribute(
        self, inputs: torch.Tensor, baseline: torch.Tensor, index: tuple[int, ...]
    ) -> FloatArray:
        self.unit.index = index
        attribution = self.deeplift.attribute(
            inputs.requires_grad_(True), baselines=baseline, target=0
        )
        self.unit.output = None
        result = attribution[0] if self.reduce is None else self.reduce(attribution[0])
        return np.asarray(result.detach().cpu().numpy())
