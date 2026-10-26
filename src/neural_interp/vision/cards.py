from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from neural_interp.vision.inception import SIZE


def receptive_box(y: int, x: int, grid: int) -> tuple[float, float, float, float]:
    step = SIZE / grid
    return x * step, y * step, (x + 1) * step, (y + 1) * step


def framed(pixels: np.ndarray, y: int, x: int, grid: int) -> Image.Image:
    image = Image.fromarray((pixels.transpose(1, 2, 0) * 255).astype(np.uint8))
    ImageDraw.Draw(image).rectangle(receptive_box(y, x, grid), outline=(255, 214, 0), width=2)
    return image


def attribution_overlay(pixels: np.ndarray, attribution: np.ndarray, y: int, x: int, grid: int) -> Image.Image:
    base = pixels.transpose(1, 2, 0) * 0.35
    scale = np.percentile(np.abs(attribution), 99.5) or 1.0
    a = np.clip(attribution / scale, -1, 1)[..., None]
    positive = np.array([1.0, 0.85, 0.2]) * np.clip(a, 0, 1)
    negative = np.array([0.3, 0.6, 1.0]) * np.clip(-a, 0, 1)
    blended = np.clip(base + positive + negative, 0, 1)
    image = Image.fromarray((blended * 255).astype(np.uint8))
    ImageDraw.Draw(image).rectangle(receptive_box(y, x, grid), outline=(255, 255, 255), width=1)
    return image


def save_tiles(out: Path, name: str, image: Image.Image, overlay: Image.Image) -> tuple[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / f"{name}.jpg", quality=88)
    overlay.save(out / f"{name}_deeplift.jpg", quality=88)
    return f"{out.name}/{name}.jpg", f"{out.name}/{name}_deeplift.jpg"
