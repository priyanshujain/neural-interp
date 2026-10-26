from typing import Any

import numpy as np
import plotly.graph_objects as go

from neural_interp.report._context import ReportContext
from neural_interp.report._render import embed
from neural_interp.report._section import Section
from neural_interp.report._views import GuideEntry
from neural_interp.selection import elbow_index


class Activations(Section):
    title = "Activations and the elbow"
    guide = (
        GuideEntry(
            "Hit",
            "An input position where the unit fires strongly. Hits are the positive activations "
            "past the elbow of the sorted curve.",
        ),
    )
    template = "sections/activations.html.j2"

    def visible(self, context: ReportContext) -> bool:
        return context.all_activations is not None

    def view(self, context: ReportContext) -> dict[str, Any]:
        values = context.all_activations
        assert values is not None
        cut = float(context.activation.min())
        positive = np.sort(values[values > 0])
        n = len(positive)
        elbow = n - 1 - elbow_index(positive[::-1])
        chord = positive[0] + (positive[-1] - positive[0]) * elbow / (n - 1)
        step = max(1, n // 4000)
        idx = np.unique(np.concatenate([np.arange(0, n, step), np.arange(max(0, n - 4000), n)]))
        curve = go.Figure()
        curve.add_vrect(
            x0=elbow,
            x1=n - 1,
            fillcolor="#ffd600",
            opacity=0.08,
            line_width=0,
            annotation_text=f"hits: {n - elbow:,}",
            annotation_position="top left",
        )
        curve.add_trace(
            go.Scattergl(
                x=idx,
                y=positive[idx],
                mode="lines",
                name="positive activations, sorted",
                line=dict(color="#8ab4ff", width=2.5),
            )
        )
        curve.add_trace(
            go.Scatter(
                x=[0, n - 1],
                y=[positive[0], positive[-1]],
                mode="lines",
                name="line from first to last point",
                line=dict(color="#9aa1ae", dash="dot"),
            )
        )
        curve.add_trace(
            go.Scatter(
                x=[elbow, elbow],
                y=[positive[elbow], chord],
                mode="lines",
                name="largest gap to the line",
                line=dict(color="#ff7f50", width=3),
            )
        )
        curve.add_trace(
            go.Scatter(
                x=[elbow],
                y=[positive[elbow]],
                mode="markers",
                name=f"elbow: activation {positive[elbow]:.4g}",
                marker=dict(color="#ffd600", size=12),
            )
        )
        curve.update_xaxes(title="sorted index (low to high)")
        curve.update_yaxes(title="activation")
        curve.update_layout(legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"))

        below = int((values <= 0).sum())
        weak = int(((values > 0) & (values < cut)).sum())
        hits = int((values >= cut).sum())
        hist = go.Figure(
            go.Histogram(x=values, nbinsx=200, marker_color="#5c6b8a", name="activations")
        )
        hist.add_vrect(
            x0=cut, x1=float(values.max()), fillcolor="#ffd600", opacity=0.1, line_width=0
        )
        hist.add_vline(x=cut, line=dict(color="#ffd600", dash="dash"))
        hist.add_vline(x=0, line=dict(color="#9aa1ae", dash="dot"))
        for x, text in [
            (float(values.min()) / 2, f"≤ 0: {below:,}<br>never hits"),
            (cut / 2, f"weak: {weak:,}"),
            ((cut + float(values.max())) / 2, f"hits: {hits:,}<br>activation ≥ {cut:.4g}"),
        ]:
            hist.add_annotation(
                x=x, y=1, yref="paper", text=text, showarrow=False, font=dict(size=14, color="#fff")
            )
        hist.update_yaxes(type="log", title="number of activations (log scale)")
        hist.update_xaxes(title="activation")
        hist.update_layout(showlegend=False)
        activations = {
            "total": len(values),
            "positive": n,
            "cut": cut,
            "hits": len(context.activation),
            "curve": embed(curve, "65vh"),
            "histogram": embed(hist, "60vh"),
        }
        return {"activations": activations}
