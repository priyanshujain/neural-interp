from collections.abc import Callable, Sequence

import numpy as np
import torch
from captum.attr import DeepLift
from torch import nn

from neural_interp.typing import FloatArray


class _Unit(nn.Module):
    def __init__(self, model: nn.Module, module: str) -> None:
        super().__init__()
        self.model = model
        self.index: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        dict(model.named_modules())[module].register_forward_hook(self._grab)

    def _grab(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self.output = output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model(inputs)
        assert self.output is not None and self.index is not None
        flat = self.output.reshape(len(self.output), -1)
        index = self.index.repeat(len(flat) // len(self.index))
        return flat.gather(1, index[:, None])


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
        self,
        inputs: torch.Tensor,
        baselines: torch.Tensor,
        indices: Sequence[tuple[int, ...]],
        output_shape: tuple[int, ...],
    ) -> FloatArray:
        flat = [int(np.ravel_multi_index(i, output_shape)) for i in indices]
        self.unit.index = torch.tensor(flat, device=inputs.device)
        attribution = self.deeplift.attribute(
            inputs.requires_grad_(True), baselines=baselines, target=0
        )
        self.unit.output = None
        result = attribution if self.reduce is None else self.reduce(attribution)
        return np.asarray(result.detach().cpu().numpy())
