from typing import Any

import numpy as np
import plotly.graph_objects as go

from neural_interp.report._context import ReportContext
from neural_interp.report._render import embed
from neural_interp.report._section import Section


class Membership(Section):
    title = "Soft membership"
    template = "sections/membership.html.j2"

    def view(self, context: ReportContext) -> dict[str, Any]:
        moved = int((context.clusters != np.asarray(context.space.tree)).sum())
        hist = go.Figure(go.Histogram(x=context.entropy, nbinsx=80, marker_color="#7e57c2"))
        hist.add_vline(
            x=0.5, line=dict(color="#ffd600", dash="dash"), annotation_text="confident below"
        )
        hist.update_xaxes(title="entropy (bits)")
        hist.update_yaxes(title="hits")
        order = [int(c) for c in np.argsort(-context.sizes, kind="stable")]
        share = context.contacts / np.maximum(context.contacts.sum(1, keepdims=True), 1)
        share = share[np.ix_(order, order)]
        labels = [f"c{c}" for c in order]
        heat = go.Figure(
            go.Heatmap(
                z=share,
                x=labels,
                y=labels,
                colorscale="Viridis",
                zmin=0,
                zmax=float(np.percentile(share[share > 0], 98)),
                hovertemplate=(
                    "from %{y} to %{x}: %{z:.1%} of mutual-neighbour edges<extra></extra>"
                ),
                colorbar=dict(title="share"),
            )
        )
        heat.update_yaxes(autorange="reversed", title="cluster")
        heat.update_xaxes(title="neighbour cluster")
        membership = {
            "moved": moved,
            "entropy": embed(hist, "45vh"),
            "touching": embed(heat, "80vh"),
        }
        return {"membership": membership}
