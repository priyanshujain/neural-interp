from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lucent.modelzoo import inceptionv1
from lucent.modelzoo.inceptionv1.helper_layers import ReluLayer
from PIL import Image
from torch import nn

SIZE = 224
RESIZE = 256


@dataclass
class Unit:
    layer: str
    channel: int
    upstream: str


MIXED4E_55 = Unit(layer="mixed4e_1x1_pre_relu_conv", channel=55, upstream="mixed4d")


def load_model(device: str) -> nn.Module:
    model = inceptionv1(pretrained=True, redirected_ReLU=False)
    for name, child in list(model.named_children()):
        if isinstance(child, ReluLayer):
            setattr(model, name, nn.ReLU())
    return model.eval().to(device)


def preprocess(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    scale = RESIZE / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    left, top = (image.width - SIZE) // 2, (image.height - SIZE) // 2
    image = image.crop((left, top, left + SIZE, top + SIZE))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
    return pixels


def to_model_input(pixels: torch.Tensor) -> torch.Tensor:
    return pixels * 255 - 117


def sample_images(pool: Path, n: int, seed: int) -> list[Path]:
    paths = sorted(pool.glob("*/*.jpg"))
    rng = np.random.default_rng(seed)
    return [paths[i] for i in sorted(rng.choice(len(paths), size=min(n, len(paths)), replace=False))]


class Probe:
    def __init__(self, model: nn.Module, unit: Unit):
        self.model, self.unit = model, unit
        self.captured: dict[str, torch.Tensor] = {}
        self.handles = [
            getattr(model, name).register_forward_hook(self._keep(name))
            for name in (unit.layer, unit.upstream)
        ]
        conv = getattr(model, unit.layer)
        self.weight = conv.weight[unit.channel, :, 0, 0].detach()
        self.bias = float(conv.bias[unit.channel].detach())

    def _keep(self, name):
        def hook(_module, _inputs, output):
            self.captured[name] = output.detach()

        return hook

    @torch.no_grad()
    def run(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model(to_model_input(pixels))
        activation = self.captured[self.unit.layer][:, self.unit.channel]
        upstream = self.captured[self.unit.upstream]
        return activation, upstream

    def hadamard(self, upstream: torch.Tensor) -> torch.Tensor:
        return upstream * self.weight[None, :]


def batches(paths: list[Path], size: int):
    for start in range(0, len(paths), size):
        chunk = paths[start : start + size]
        yield start, torch.stack([preprocess(p) for p in chunk])


def collect_activations(probe: Probe, paths: list[Path], device: str, batch: int = 64) -> np.ndarray:
    out = []
    for _, pixels in batches(paths, batch):
        activation, _ = probe.run(pixels.to(device))
        out.append(activation.cpu().numpy())
    return np.concatenate(out)


def collect_hadamard(
    probe: Probe, paths: list[Path], hits: np.ndarray, grid: tuple[int, int], device: str, batch: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    image, y, x = np.unravel_index(hits, (len(paths), *grid))
    rows = np.empty((len(hits), probe.weight.numel()), dtype=np.float32)
    check = np.empty(len(hits), dtype=np.float32)
    needed = np.unique(image)
    for start in range(0, len(needed), batch):
        ids = needed[start : start + batch]
        pixels = torch.stack([preprocess(paths[i]) for i in ids]).to(device)
        activation, upstream = probe.run(pixels)
        local = {int(i): j for j, i in enumerate(ids)}
        for h in np.flatnonzero(np.isin(image, ids)):
            j = local[int(image[h])]
            contribution = probe.hadamard(upstream[j : j + 1, :, y[h], x[h]])[0]
            rows[h] = contribution.cpu().numpy()
            check[h] = float(activation[j, y[h], x[h]]) - probe.bias
    return rows, check


class NeuronAttribution(nn.Module):
    def __init__(self, model: nn.Module, unit: Unit):
        super().__init__()
        from captum.attr import DeepLift

        self.model, self.unit = model, unit
        self.position = (0, 0)
        self.out: torch.Tensor | None = None
        getattr(model, unit.layer).register_forward_hook(self._grab)
        self.deeplift = DeepLift(self)

    def _grab(self, _module, _inputs, output):
        self.out = output

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        self.model(inp)
        y, x = self.position
        return self.out[:, self.unit.channel, y, x][:, None]

    def map(self, pixels: torch.Tensor, y: int, x: int) -> np.ndarray:
        self.position = (y, x)
        inp = to_model_input(pixels[None]).requires_grad_(True)
        baseline = to_model_input(torch.zeros_like(pixels[None]))
        attr = self.deeplift.attribute(inp, baselines=baseline, target=0)
        self.out = None
        return attr[0].sum(0).detach().cpu().numpy()
