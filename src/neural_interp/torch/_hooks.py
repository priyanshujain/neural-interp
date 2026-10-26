from collections.abc import Callable
from types import TracebackType
from typing import Self

import torch
from torch import nn


class Capture:
    def __init__(self, model: nn.Module, modules: list[str]) -> None:
        self.model = model
        self.modules = modules
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _keep(
        self, name: str
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(
            _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            self.outputs[name] = output.detach()

        return hook

    def __enter__(self) -> Self:
        named = dict(self.model.named_modules())
        self.handles = [
            named[name].register_forward_hook(self._keep(name)) for name in self.modules
        ]
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.no_grad()
    def run(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.outputs = {}
        self.model(inputs)
        return self.outputs
