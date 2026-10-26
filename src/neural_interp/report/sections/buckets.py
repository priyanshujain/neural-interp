from typing import Any

import numpy as np

from neural_interp.buckets import ranked_sample
from neural_interp.report._context import LinkedBucket, ReportContext, ref
from neural_interp.report._render import embed, render
from neural_interp.report._section import Section
from neural_interp.report._views import BucketRow, TileView
from neural_interp.report.sections.map import map_figure


class BucketPages(Section):
    title = "All buckets"
    template = "sections/buckets.html.j2"

    def __init__(self, stars: int = 8, tiles: int = 6) -> None:
        self.stars = stars
        self.tiles = tiles

    def sample(self, context: ReportContext, linked: LinkedBucket, n: int) -> list[int]:
        return [int(h) for h in ranked_sample(linked.bucket, context.probabilities, n)]

    def hits(self, context: ReportContext) -> set[int]:
        return {h for b in context.buckets for h in self.sample(context, b, self.tiles)}

    def tile(self, context: ReportContext, number: int, hit: int) -> TileView:
        p = context.probabilities[hit]
        return TileView(
            number=number,
            id=context.space.hit_ids[hit],
            media=context.hit_view.media(hit),
            label=context.hit_view.label(hit),
            activation=float(context.activation[hit]),
            entropy=float(context.entropy[hit]),
            pattern=context.pattern(hit),
            probabilities=[(ref(c), float(p[c])) for c in np.argsort(-p)[:3]],
        )

    def page(self, context: ReportContext, linked: LinkedBucket) -> str:
        b = linked.bucket
        stars = self.sample(context, linked, self.stars)
        shown = self.sample(context, linked, self.tiles)
        figure = map_figure(context, context.positions, "cluster", stars, shown)
        return render(
            "bucket.html.j2",
            name=b.name,
            clusters=[ref(c) for c in b.clusters],
            size=len(b.hits),
            single=len(b.clusters) == 1,
            threshold=b.threshold,
            composition=context.composition(np.asarray(b.hits)),
            map=embed(figure, "80vh"),
            tiles=[self.tile(context, i + 1, h) for i, h in enumerate(shown)],
        )

    def write_assets(self, context: ReportContext) -> None:
        for linked in context.buckets:
            (context.out / linked.href).write_text(self.page(context, linked))

    def view(self, context: ReportContext) -> dict[str, Any]:
        rows = [BucketRow(b.bucket.name, b.href, len(b.bucket.hits)) for b in context.buckets]
        return {"buckets": rows}
