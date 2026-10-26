from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from markupsafe import Markup

from neural_interp.report._context import ReportContext
from neural_interp.report._render import color, embed
from neural_interp.report._section import Section
from neural_interp.report._views import GuideEntry, TreePick, UpClose


class Forest(Section):
    title = "Density-peak forest"
    guide = (
        GuideEntry(
            "Distance",
            "Weighted Jaccard between hit vectors, positive and negative parts compared "
            "separately. 0 means the same inputs drove both hits in the same amounts, 1 means no "
            "input in common.",
        ),
        GuideEntry(
            "Forest and clusters",
            "Each hit links to its nearest denser hit. Hits that are both dense and far from "
            "anything denser become roots, and every other hit climbs to one root. Each root's "
            "tree is a cluster.",
        ),
    )
    template = "sections/forest.html.j2"

    def peak_separation(self, context: ReportContext) -> Markup:
        f = context.space.forest
        is_root = np.zeros(len(f.rho_), bool)
        is_root[f.roots_] = True
        candidates = np.flatnonzero(~is_root & (f.rho_ >= f.rho_cut_))
        runners = candidates[np.argsort(-f.delta_[candidates])][:10]
        tree_of_root = {int(r): t for t, r in enumerate(f.roots_)}
        shown = sorted(
            [int(r) for r in f.roots_] + [int(r) for r in runners], key=lambda i: -f.delta_[i]
        )
        fig = go.Figure(
            go.Bar(
                x=[f"c{tree_of_root[i]}" if i in tree_of_root else f"#{i}" for i in shown],
                y=[f.delta_[i] for i in shown],
                marker_color=[
                    color(tree_of_root[i]) if i in tree_of_root else "#4b5060" for i in shown
                ],
                text=[f"rho {f.rho_[i]:.3f}" for i in shown],
                textposition="none",
                hovertemplate="%{x}<br>delta %{y:.3f}<br>%{text}<extra></extra>",
            )
        )
        fig.add_hline(
            y=f.delta_cut_, line=dict(color="#ffd600", dash="dash"), annotation_text="delta cut"
        )
        fig.update_yaxes(
            title="delta: distance to the nearest denser point",
            range=[f.delta_cut_ * 0.8, float(f.delta_[shown].max()) * 1.03],
        )
        fig.update_xaxes(
            title="roots (coloured) and the best non-roots that are dense enough (grey)",
            type="category",
        )
        return embed(fig, "60vh")

    def forest_up_close(
        self, context: ReportContext, per_tree: int = 60, iterations: int = 150, seed: int = 0
    ) -> UpClose:
        f = context.space.forest
        point_mode = np.asarray(context.space.membership.probabilities_).argmax(1)
        n_trees = len(f.roots_)
        rng = np.random.default_rng(seed)
        chosen: set[int] = set()
        for t in range(n_trees):
            members = np.flatnonzero(f.labels_ == t)
            for i in rng.choice(members, min(per_tree, len(members)), replace=False):
                node = int(i)
                while node >= 0 and node not in chosen:
                    chosen.add(node)
                    node = int(f.parent_[node])
        nodes = np.array(sorted(chosen))
        index = {int(n): k for k, n in enumerate(nodes)}
        dist = np.asarray(context.space.distances)[np.ix_(nodes, nodes)]
        g = nx.Graph()
        g.add_nodes_from(range(len(nodes)))
        masked = dist + np.diag(np.full(len(nodes), np.inf))
        for i, row in enumerate(np.argsort(masked, axis=1)[:, :6]):
            for j in row:
                g.add_edge(i, int(j), weight=float(1 - dist[i, j]))
        for k, n in enumerate(nodes):
            if f.parent_[n] >= 0:
                g.add_edge(k, index[int(f.parent_[n])], weight=2.0)
        pos = nx.spring_layout(g, weight="weight", iterations=iterations, seed=seed)
        xy = np.array([pos[k] for k in range(len(nodes))])
        rho = f.rho_[nodes]
        size = 4 + 14 * (rho - rho.min()) / max(np.ptp(rho), 1e-9)

        counts = context.contacts.copy()
        np.fill_diagonal(counts, 0)
        big = np.argsort(-context.sizes)[:8]
        sub = counts[np.ix_(big, big)]
        default = {int(big[i]) for i in np.unravel_index(np.argmax(sub), sub.shape)}

        fig = go.Figure()
        for t in range(n_trees):
            on = t in default
            kids = [k for k, n in enumerate(nodes) if f.labels_[n] == t and f.parent_[n] >= 0]
            parents = [index[int(f.parent_[nodes[k]])] for k in kids]
            gap = np.full(len(kids), np.nan)
            fig.add_trace(
                go.Scatter(
                    x=np.column_stack([xy[kids, 0], xy[parents, 0], gap]).ravel(),
                    y=np.column_stack([xy[kids, 1], xy[parents, 1], gap]).ravel(),
                    mode="lines",
                    line=dict(color=color(t), width=1),
                    opacity=0.6,
                    hoverinfo="skip",
                    showlegend=False,
                    visible=on,
                )
            )
            sel = np.array([k for k, n in enumerate(nodes) if f.labels_[n] == t])
            fig.add_trace(
                go.Scatter(
                    x=xy[sel, 0],
                    y=xy[sel, 1],
                    mode="markers",
                    name=f"c{t}",
                    visible=on,
                    marker=dict(size=size[sel], color=color(t), line=dict(color="#000", width=0.5)),
                    text=[
                        f"tree c{t} | depth {f.depth_[nodes[k]]} | rho {rho[k]:.3f} | most "
                        f"probable c{point_mode[nodes[k]]}"
                        for k in sel
                    ],
                    hoverinfo="text",
                )
            )
            r = index[int(f.roots_[t])]
            fig.add_trace(
                go.Scatter(
                    x=[xy[r, 0]],
                    y=[xy[r, 1]],
                    mode="markers+text",
                    text=[f"c{t}"],
                    textposition="top center",
                    textfont=dict(size=15, color="#fff"),
                    showlegend=False,
                    visible=on,
                    hoverinfo="skip",
                    marker=dict(
                        symbol="star", size=22, color=color(t), line=dict(color="#fff", width=1.5)
                    ),
                )
            )
        fig.update_xaxes(visible=False, autorange=True)
        fig.update_yaxes(visible=False, autorange=True, scaleanchor="x")
        fig.update_layout(showlegend=False)

        order = [
            int(c) for c in np.argsort(-np.bincount(f.labels_, minlength=n_trees), kind="stable")
        ]
        picks = [TreePick(t, color(t), t in default) for t in order]
        return UpClose(per_tree, picks, embed(fig, "80vh", div_id="upclose"))

    def forest_order(self, context: ReportContext) -> np.ndarray:
        f = context.space.forest
        children: dict[int, list[int]] = {}
        for i, p in enumerate(f.parent_):
            if p >= 0:
                children.setdefault(int(p), []).append(i)
        order: list[int] = []
        for root in f.roots_:
            stack = [int(root)]
            while stack:
                node = stack.pop()
                order.append(node)
                stack.extend(sorted(children.get(node, []), key=lambda k: f.rho_[k]))
        return np.array(order)

    def view(self, context: ReportContext) -> dict[str, Any]:
        f = context.space.forest
        counts = np.asarray(context.space.counts)
        point_cluster = np.asarray(context.space.membership.probabilities_).argmax(1)
        n_roots = len(f.roots_)
        dg = go.Figure()
        dg.add_trace(
            go.Scattergl(
                x=f.rho_,
                y=f.delta_,
                mode="markers",
                name="hits",
                marker=dict(size=4, color=[color(c) for c in f.labels_], opacity=0.5),
                text=[f"tree c{t}" for t in f.labels_],
                hoverinfo="text+x+y",
            )
        )
        dg.add_trace(
            go.Scatter(
                x=f.rho_[f.roots_],
                y=f.delta_[f.roots_],
                mode="markers+text",
                name="roots",
                text=[f"c{t}" for t in range(n_roots)],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=[color(t) for t in range(n_roots)],
                    line=dict(color="#fff", width=1.5),
                ),
            )
        )
        dg.add_vline(
            x=f.rho_cut_, line=dict(color="#ffd600", dash="dash"), annotation_text="rho cut"
        )
        dg.add_hline(
            y=f.delta_cut_, line=dict(color="#ffd600", dash="dash"), annotation_text="delta cut"
        )
        dg.update_xaxes(title="rho (log density)")
        dg.update_yaxes(title="delta (distance to nearest denser point)")

        order = self.forest_order(context)
        x = np.empty(len(order))
        x[order] = np.arange(len(order))
        land = go.Figure()
        for t in range(n_roots):
            members = np.flatnonzero(f.labels_ == t)
            linked = members[f.parent_[members] >= 0]
            gap = np.full(len(linked), np.nan)
            land.add_trace(
                go.Scattergl(
                    x=np.column_stack([x[linked], x[f.parent_[linked]], gap]).ravel(),
                    y=np.column_stack([f.rho_[linked], f.rho_[f.parent_[linked]], gap]).ravel(),
                    mode="lines",
                    line=dict(color=color(t), width=0.7),
                    opacity=0.5,
                    legendgroup=f"c{t}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            land.add_trace(
                go.Scattergl(
                    x=x[members],
                    y=f.rho_[members],
                    mode="markers",
                    name=f"c{t} ({int(counts[members].sum()):,})",
                    legendgroup=f"c{t}",
                    marker=dict(size=3, color=color(t)),
                    text=[
                        f"tree c{t} | depth {f.depth_[i]} | most probable c{point_cluster[i]}"
                        for i in members
                    ],
                    hoverinfo="text",
                )
            )
        land.add_trace(
            go.Scatter(
                x=x[f.roots_],
                y=f.rho_[f.roots_],
                mode="markers+text",
                name="roots",
                text=[f"c{t}" for t in range(n_roots)],
                textposition="top center",
                marker=dict(size=11, symbol="triangle-up", color="#fff"),
            )
        )
        land.update_xaxes(title="points, ordered along their tree", showticklabels=False)
        land.update_yaxes(title="rho (log density)")
        land.update_layout(
            legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=11))
        )
        forest = {
            "rho_cut": f.rho_cut_,
            "delta_cut": f.delta_cut_,
            "n_roots": n_roots,
            "decision": embed(dg, "70vh"),
            "separation": self.peak_separation(context),
            "up_close": self.forest_up_close(context),
            "landscape": embed(land, "75vh"),
        }
        return {"forest": forest}
