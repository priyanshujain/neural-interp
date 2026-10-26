from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from markupsafe import Markup

from neural_interp.report._render import render
from neural_interp.report._views import GuideEntry
from neural_interp.space import HitSpace


class HitView(Protocol):
    def media(self, hit: int) -> Markup: ...

    def strip(self, hits: list[int]) -> Markup: ...

    def label(self, hit: int) -> str: ...

    def hover_extras(self, hit: int) -> str: ...

    def guide(self) -> list[GuideEntry]: ...


@dataclass
class Tile:
    image: str
    overlay: str
    label: str


class ImageHitView:
    def __init__(self, space: HitSpace, tile: Callable[[int], Tile]) -> None:
        self.space = space
        self.tile_of = tile
        self.tiles: dict[int, Tile] = {}
        self.activation = np.asarray(space.activation)

    def tile(self, hit: int) -> Tile:
        if hit not in self.tiles:
            self.tiles[hit] = self.tile_of(hit)
        return self.tiles[hit]

    def media(self, hit: int) -> Markup:
        return Markup(render("hits/image_media.html.j2", tile=self.tile(hit)))

    def strip(self, hits: list[int]) -> Markup:
        cells = [(self.space.hit_ids[h], self.tile(h), float(self.activation[h])) for h in hits]
        return Markup(render("hits/image_strip.html.j2", cells=cells))

    def label(self, hit: int) -> str:
        return self.tile(hit).label

    def hover_extras(self, hit: int) -> str:
        return ""

    def guide(self) -> list[GuideEntry]:
        return [
            GuideEntry(
                "Hit, for an image unit",
                "One image patch: the unit's value at one position of its feature map. Cards show "
                "the input with that position boxed.",
            ),
            GuideEntry(
                "DeepLift",
                "Which input pixels drove this hit, against a black baseline. Yellow raises the "
                "unit, blue lowers it. The box marks the hit's position.",
            ),
        ]
