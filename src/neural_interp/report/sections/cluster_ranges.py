from typing import Any

import plotly.graph_objects as go

from neural_interp.report._context import ReportContext
from neural_interp.report._render import color, embed
from neural_interp.report._section import Section


class ClusterRanges(Section):
    title = "Activation range of each cluster"
    template = "sections/cluster_ranges.html.j2"

    def view(self, context: ReportContext) -> dict[str, Any]:
        fig = go.Figure()
        for c in context.clusters_by_size():
            members = context.clusters == c
            if members.any():
                fig.add_trace(
                    go.Box(
                        y=context.activation[members],
                        name=f"c{c}",
                        marker_color=color(c),
                        boxpoints="outliers",
                        line_width=1,
                        marker_size=3,
                    )
                )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="activation", type="log")
        return {"cluster_ranges": embed(fig, "55vh")}
