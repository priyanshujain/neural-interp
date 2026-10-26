from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from markupsafe import Markup

from neural_interp.buckets import Bucket
from neural_interp.report._context import Positions, ReportContext, link_buckets
from neural_interp.report._hits import HitView, ImageHitView, Tile
from neural_interp.report._render import render
from neural_interp.report._section import Section
from neural_interp.report._views import GuideItem
from neural_interp.report.sections import (
    Activations,
    BucketPages,
    ClusterCards,
    ClusterRanges,
    Forest,
    GroupCards,
    Map,
    Membership,
)
from neural_interp.space import HitSpace
from neural_interp.typing import FloatArray


def default_sections() -> list[Section]:
    return [
        Activations(),
        ClusterRanges(),
        Forest(),
        Membership(),
        Map(),
        ClusterCards(),
        GroupCards(),
        BucketPages(),
    ]


class Report:
    def __init__(
        self,
        space: HitSpace,
        hit_view: HitView,
        sections: Sequence[Section] | None = None,
        group_of: Callable[[int], str] | None = None,
        group_colors: dict[str, str] | None = None,
        all_activations: FloatArray | None = None,
        unit: str = "unit",
    ) -> None:
        self.sections = list(sections) if sections is not None else default_sections()
        self.context = ReportContext.build(
            space, hit_view, group_of, group_colors, all_activations, unit
        )

    def with_buckets(
        self, buckets: list[Bucket], out: Path = Path("."), positions: Positions | None = None
    ) -> ReportContext:
        return replace(
            self.context, buckets=link_buckets(buckets), out=out, positions=positions or {}
        )

    def hits_to_render(self, buckets: list[Bucket]) -> list[int]:
        context = self.with_buckets(buckets)
        wanted: set[int] = set()
        for section in self.sections:
            if section.visible(context):
                wanted |= section.hits(context)
        return sorted(wanted)

    def header_stats(self) -> list[tuple[str, str]]:
        c = self.context
        return [
            (
                f"{len(c.all_activations):,}" if c.all_activations is not None else "-",
                "activations",
            ),
            (f"{len(c.space.hit_ids):,}", "hits"),
            (f"{c.n_clusters}", "clusters"),
            (f"{int((c.entropy < 0.5).sum()):,}", "confident hits"),
            (f"{len(c.space.membership.touching_)}", "touching cluster pairs"),
        ]

    def write(
        self, out: Path, buckets: list[Bucket], positions: dict[int, tuple[float, float]]
    ) -> None:
        (out / "patterns").mkdir(parents=True, exist_ok=True)
        clean = {int(k): (float(v[0]), float(v[1])) for k, v in positions.items()}
        context = self.with_buckets(buckets, out, clean)
        shown = [s for s in self.sections if s.visible(context)]
        for section in shown:
            section.write_assets(context)
        bodies: list[Markup] = [s.render(context, i + 1) for i, s in enumerate(shown)]
        guide = [
            GuideItem(e.term, e.text, i + 1, s.title) for i, s in enumerate(shown) for e in s.guide
        ] + [GuideItem(e.term, e.text, None, "") for e in context.hit_view.guide()]
        page = render(
            "index.html.j2",
            unit=context.unit,
            stats=self.header_stats(),
            guide=guide,
            sections=bodies,
        )
        (out / "index.html").write_text(page)


class HitSpaceReport(Report):
    def __init__(
        self,
        space: HitSpace,
        tile: Callable[[int], Tile],
        group_of: Callable[[int], str] | None = None,
        group_colors: dict[str, str] | None = None,
        all_activations: FloatArray | None = None,
        unit: str = "unit",
    ) -> None:
        super().__init__(
            space,
            ImageHitView(space, tile),
            default_sections(),
            group_of,
            group_colors,
            all_activations,
            unit,
        )
