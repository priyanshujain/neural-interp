import html
from typing import Any, Callable

import numpy as np
import plotly.graph_objects as go

from neural_interp.report._context import Positions, ReportContext
from neural_interp.report._render import color, embed
from neural_interp.report._section import Section


def hover(context: ReportContext, hit: int) -> str:
    p = context.probabilities[hit]
    probs = " ".join(f"c{c} {p[c]:.2f}" for c in np.argsort(-p)[:3] if p[c] >= 0.05)
    group = f" | {context.groups[hit]}" if context.groups is not None else ""
    extras = context.hit_view.hover_extras(hit)
    return (
        f"{html.escape(context.space.hit_ids[hit])}<br>act "
        f"{context.activation[hit]:.4g}{group}<br>{probs}" + (f"<br>{extras}" if extras else "")
    )


def map_figure(
    context: ReportContext,
    positions: Positions,
    by: str,
    stars: list[int] | None = None,
    numbered: list[int] | None = None,
) -> go.Figure:
    nodes = np.array(list(positions))
    xy = np.array([positions[n] for n in nodes])
    name: Callable[[Any], str]
    col: Callable[[Any], str]
    if by == "cluster":
        labels = context.clusters[nodes]
        keys = sorted(set(labels.tolist()), key=lambda c: -context.sizes[c])
        name, col = (lambda k: f"c{k}"), color
    else:
        assert context.groups is not None
        labels = context.groups[nodes]
        keys = sorted(set(labels.tolist()))
        name, col = (lambda k: k), context.group_color
    faded = stars is not None
    fig = go.Figure()
    for k in keys:
        sel = labels == k
        fig.add_trace(
            go.Scattergl(
                x=xy[sel, 0],
                y=xy[sel, 1],
                mode="markers",
                name=name(k),
                marker=dict(size=7, color=col(k), opacity=0.3 if faded else 0.85),
                text=[hover(context, h) for h in nodes[sel]],
                hoverinfo="text",
            )
        )
    if stars is not None:
        shown = [h for h in stars if h in positions]
        fig.add_trace(
            go.Scatter(
                x=[positions[h][0] for h in shown],
                y=[positions[h][1] for h in shown],
                mode="markers",
                name="sampled",
                marker=dict(
                    symbol="star", size=17, color="#ffd600", line=dict(color="#000", width=1)
                ),
                text=[hover(context, h) for h in shown],
                hoverinfo="text",
            )
        )
    if numbered is not None:
        labelled = [(i + 1, h) for i, h in enumerate(numbered) if h in positions]
        fig.add_trace(
            go.Scatter(
                x=[positions[h][0] for _, h in labelled],
                y=[positions[h][1] for _, h in labelled],
                mode="text",
                text=[f"<b>{i}</b>" for i, _ in labelled],
                textposition="top center",
                textfont=dict(color="#fff", size=16),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(
        legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=11))
    )
    return fig


class Map(Section):
    title = "Map of the hit space"
    template = "sections/map.html.j2"

    def view(self, context: ReportContext) -> dict[str, Any]:
        positions = context.positions
        maps = [("By cluster", embed(map_figure(context, positions, "cluster"), "85vh"))]
        if context.groups is not None:
            maps.append(("By class", embed(map_figure(context, positions, "group"), "85vh")))
        return {"maps": maps}
