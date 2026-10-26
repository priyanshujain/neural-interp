from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from neural_interp.vision.inception import SIZE


def overlay(pixels: np.ndarray, attribution: np.ndarray, y: int, x: int, grid: int, alpha: float = 0.6) -> Image.Image:
    base = (pixels.transpose(1, 2, 0) * 255).astype(np.uint8)
    scale = np.percentile(np.abs(attribution), 99.5) or 1.0
    a = np.clip(attribution / scale, -1, 1)
    heat = np.zeros((*a.shape, 3), dtype=np.float32)
    heat[..., 0] = np.clip(a, 0, 1)
    heat[..., 2] = np.clip(-a, 0, 1)
    strength = np.abs(a)[..., None] * alpha
    blended = base * (1 - strength) + heat * 255 * strength
    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    step = SIZE / grid
    box = (x * step, y * step, (x + 1) * step, (y + 1) * step)
    ImageDraw.Draw(image).rectangle(box, outline=(255, 220, 0), width=2)
    return image


def save_card(out: Path, name: str, original: np.ndarray, attributed: Image.Image) -> str:
    out.mkdir(parents=True, exist_ok=True)
    left = Image.fromarray((original.transpose(1, 2, 0) * 255).astype(np.uint8))
    sheet = Image.new("RGB", (SIZE * 2, SIZE))
    sheet.paste(left, (0, 0))
    sheet.paste(attributed, (SIZE, 0))
    sheet.save(out / f"{name}.png")
    return f"{out.name}/{name}.png"
