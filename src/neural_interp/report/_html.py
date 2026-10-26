import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from markupsafe import Markup
from PIL import Image

from neural_interp.buckets import Bucket, ranked_sample
from neural_interp.cluster import knn_graph
from neural_interp.report._render import PALETTE, color, embed, render
from neural_interp.report._views import (
    ClusterCard,
    ClusterRef,
    GroupCard,
    HitView,
    Part,
    TileView,
    TreePick,
    UpClose,
)
from neural_interp.selection import elbow_index
from neural_interp.space import HitSpace
from neural_interp.typing import FloatArray

Positions = dict[int, tuple[float, float]]
BucketLinks = dict[int, list[tuple[float, str]]]


@dataclass
class Tile:
    image: str
    overlay: str
    label: str


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "bucket"


def ref(cluster: int) -> ClusterRef:
    return ClusterRef(int(cluster), color(cluster))


def source_of(hit_id: str) -> str:
    return hit_id.rsplit(":", 1)[0]


def pattern_png(row: np.ndarray, path: Path, cols: int = 24, cell: int = 6) -> None:
    n = len(row)
    rows = -(-n // cols)
    grid = np.zeros(rows * cols)
    grid[:n] = row
    scale = np.percentile(np.abs(grid[:n]), 99) or 1.0
    v = grid.reshape(rows, cols) / scale
    v = np.sign(v) * np.sqrt(np.clip(np.abs(v), 0, 1))
    base = np.array([0.15, 0.16, 0.19])
    rgb = base + np.clip(v, 0, 1)[..., None] * (np.array([0.35, 0.95, 0.4]) - base)
    rgb = rgb + np.clip(-v, 0, 1)[..., None] * (np.array([0.95, 0.3, 0.3]) - base)
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    image.resize((cols * cell, rows * cell), Image.Resampling.NEAREST).save(path)


class HitSpaceReport:
    def __init__(
        self,
        space: HitSpace,
        tile: Callable[[int], Tile],
        group_of: Callable[[int], str] | None = None,
        group_colors: dict[str, str] | None = None,
        all_activations: FloatArray | None = None,
        unit: str = "unit",
    ) -> None:
        self.out = Path(".")
        self.space, self.tile_of, self.unit = space, tile, unit
        self.all_activations = all_activations
        self.probs = np.asarray(space.probabilities)
        self.cluster = np.asarray(space.cluster)
        self.activation = np.asarray(space.activation)
        self.entropy = np.asarray(space.entropy)
        self.n_clusters = self.probs.shape[1]
        self.sizes = np.bincount(self.cluster, minlength=self.n_clusters)
        self.tiles: dict[int, Tile] = {}
        self.groups = (
            np.array([group_of(h) for h in range(len(space.hit_ids))]) if group_of else None
        )
        names = sorted(set(self.groups.tolist())) if self.groups is not None else []
        self.group_colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)} | (
            group_colors or {}
        )
        self.contacts = self.contact_counts()

    def group_color(self, name: str) -> str:
        return self.group_colors.get(name, "#aaaaaa")

    def contact_counts(self) -> np.ndarray:
        modes = np.asarray(self.space.membership.probabilities_).argmax(1)
        edges = knn_graph(np.asarray(self.space.distances), 5, mutual=True).tocoo()
        counts = np.zeros((self.n_clusters, self.n_clusters))
        np.add.at(counts, (modes[edges.row], modes[edges.col]), 1)
        return counts

    def neighbours(self, c: int, n: int = 5) -> list[tuple[int, float]]:
        row = self.contacts[c].copy()
        row[c] = 0
        total = row.sum()
        return (
            [(int(k), row[k] / total) for k in np.argsort(-row)[:n] if row[k] > 0] if total else []
        )

    def tile(self, hit: int) -> Tile:
        if hit not in self.tiles:
            self.tiles[hit] = self.tile_of(hit)
        return self.tiles[hit]

    def pattern(self, hit: int) -> str:
        path = self.out / "patterns" / f"hit_{hit}.png"
        if not path.exists():
            pattern_png(np.asarray(self.space.rows[hit]), path)
        return f"patterns/{path.name}"

    def cluster_pattern(self, c: int) -> str:
        path = self.out / "patterns" / f"cluster_{c}.png"
        if not path.exists():
            pattern_png(np.asarray(self.space.rows[self.cluster == c]).mean(0), path)
        return f"patterns/{path.name}"

    def header_stats(self) -> list[tuple[str, str]]:
        return [
            (
                f"{len(self.all_activations):,}" if self.all_activations is not None else "-",
                "activations",
            ),
            (f"{len(self.space.hit_ids):,}", "hits"),
            (f"{self.n_clusters}", "clusters"),
            (f"{int((self.entropy < 0.5).sum()):,}", "confident hits"),
            (f"{len(self.space.membership.touching_)}", "touching cluster pairs"),
        ]

    def activations(self) -> dict[str, Any]:
        values = self.all_activations
        assert values is not None
        cut = float(self.activation.min())
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
        return {
            "total": len(values),
            "positive": n,
            "cut": cut,
            "hits": len(self.activation),
            "curve": embed(curve, "65vh"),
            "histogram": embed(hist, "60vh"),
        }

    def hit_ranges(self) -> Markup:
        fig = go.Figure()
        for c in np.argsort(-self.sizes, kind="stable"):
            members = self.cluster == c
            if members.any():
                fig.add_trace(
                    go.Box(
                        y=self.activation[members],
                        name=f"c{c}",
                        marker_color=color(int(c)),
                        boxpoints="outliers",
                        line_width=1,
                        marker_size=3,
                    )
                )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="activation", type="log")
        return embed(fig, "55vh")

    def peak_separation(self) -> Markup:
        f = self.space.forest
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

    def forest_up_close(self, per_tree: int = 60, iterations: int = 150, seed: int = 0) -> UpClose:
        f = self.space.forest
        point_mode = np.asarray(self.space.membership.probabilities_).argmax(1)
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
        dist = np.asarray(self.space.distances)[np.ix_(nodes, nodes)]
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

        counts = self.contacts.copy()
        np.fill_diagonal(counts, 0)
        big = np.argsort(-self.sizes)[:8]
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

    def forest_order(self) -> np.ndarray:
        f = self.space.forest
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

    def forest(self) -> dict[str, Any]:
        f = self.space.forest
        counts = np.asarray(self.space.counts)
        point_cluster = np.asarray(self.space.membership.probabilities_).argmax(1)
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

        order = self.forest_order()
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
        return {
            "rho_cut": f.rho_cut_,
            "delta_cut": f.delta_cut_,
            "n_roots": n_roots,
            "decision": embed(dg, "70vh"),
            "separation": self.peak_separation(),
            "up_close": self.forest_up_close(),
            "landscape": embed(land, "75vh"),
        }

    def membership(self) -> dict[str, Any]:
        moved = int((self.cluster != np.asarray(self.space.tree)).sum())
        hist = go.Figure(go.Histogram(x=self.entropy, nbinsx=80, marker_color="#7e57c2"))
        hist.add_vline(
            x=0.5, line=dict(color="#ffd600", dash="dash"), annotation_text="confident below"
        )
        hist.update_xaxes(title="entropy (bits)")
        hist.update_yaxes(title="hits")
        order = [int(c) for c in np.argsort(-self.sizes, kind="stable")]
        share = self.contacts / np.maximum(self.contacts.sum(1, keepdims=True), 1)
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
        return {"moved": moved, "entropy": embed(hist, "45vh"), "touching": embed(heat, "80vh")}

    def hover(self, hit: int) -> str:
        p = self.probs[hit]
        probs = " ".join(f"c{c} {p[c]:.2f}" for c in np.argsort(-p)[:3] if p[c] >= 0.05)
        group = f" | {self.groups[hit]}" if self.groups is not None else ""
        return (
            f"{html.escape(self.space.hit_ids[hit])}<br>act "
            f"{self.activation[hit]:.4g}{group}<br>{probs}"
        )

    def map_figure(
        self,
        positions: Positions,
        by: str,
        stars: list[int] | None = None,
        numbered: list[int] | None = None,
    ) -> go.Figure:
        nodes = np.array(list(positions))
        xy = np.array([positions[n] for n in nodes])
        if by == "cluster":
            labels = self.cluster[nodes]
            keys = sorted(set(labels.tolist()), key=lambda c: -self.sizes[c])
            name, col = (lambda k: f"c{k}"), color
        else:
            assert self.groups is not None
            labels = self.groups[nodes]
            keys = sorted(set(labels.tolist()))
            name, col = (lambda k: k), self.group_color
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
                    text=[self.hover(h) for h in nodes[sel]],
                    hoverinfo="text",
                )
            )
        if faded:
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
                    text=[self.hover(h) for h in shown],
                    hoverinfo="text",
                )
            )
        if numbered is not None:
            shown = [(i + 1, h) for i, h in enumerate(numbered) if h in positions]
            fig.add_trace(
                go.Scatter(
                    x=[positions[h][0] for _, h in shown],
                    y=[positions[h][1] for _, h in shown],
                    mode="text",
                    text=[f"<b>{i}</b>" for i, _ in shown],
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

    def maps(self, positions: Positions) -> list[tuple[str, Markup]]:
        panes = [("By cluster", embed(self.map_figure(positions, "cluster"), "85vh"))]
        if self.groups is not None:
            panes.append(("By class", embed(self.map_figure(positions, "group"), "85vh")))
        return panes

    def composition(self, members: np.ndarray) -> list[Part]:
        if self.groups is None or not len(members):
            return []
        names, counts = np.unique(self.groups[members], return_counts=True)
        share = counts / counts.sum()
        return [Part(str(n), float(s), self.group_color(str(n))) for n, s in zip(names, share)]

    def hit_views(self, hits: list[int]) -> list[HitView]:
        views: list[HitView] = []
        for h in hits:
            t = self.tile(h)
            views.append(
                HitView(
                    self.space.hit_ids[h],
                    t.image,
                    t.overlay,
                    t.label,
                    float(self.activation[h]),
                    self.pattern(h),
                )
            )
        return views

    def hits_to_render(self, buckets: list[Bucket]) -> list[int]:
        wanted: set[int] = set()
        for c in range(self.n_clusters):
            if self.sizes[c]:
                wanted.update(self.core_hits(c))
        for b in buckets:
            wanted.update(int(h) for h in ranked_sample(b, self.probs, 6))
            if len(b.clusters) > 1:
                wanted.update(int(h) for h in ranked_sample(b, self.probs, 10))
        return sorted(wanted)

    def core_hits(self, c: int, n: int = 10) -> list[int]:
        members = np.flatnonzero(self.cluster == c)
        members = members[np.argsort(-self.probs[members, c], kind="stable")]
        picked, seen = [], set()
        for h in members:
            key = source_of(self.space.hit_ids[h])
            if key not in seen:
                seen.add(key)
                picked.append(int(h))
                if len(picked) == n:
                    break
        return picked

    def cluster_card(self, c: int, bucket_links: BucketLinks) -> ClusterCard:
        members = np.flatnonzero(self.cluster == c)
        acts = self.activation[members]
        return ClusterCard(
            cluster=ref(c),
            size=len(members),
            sources=len({source_of(self.space.hit_ids[h]) for h in members}),
            composition=self.composition(members),
            low=float(acts.min()),
            high=float(acts.max()),
            median=float(np.median(acts)),
            confident=float((self.entropy[members] < 0.5).mean()),
            self_contained=float(self.contacts[c, c] / max(self.contacts[c].sum(), 1)),
            neighbours=[(ref(k), float(s)) for k, s in self.neighbours(c)],
            pattern=self.cluster_pattern(c),
            bucket_links=bucket_links.get(c, []),
            hits=self.hit_views(self.core_hits(c)),
        )

    def cluster_cards(self, bucket_links: BucketLinks) -> list[ClusterCard]:
        order = [int(c) for c in np.argsort(-self.sizes, kind="stable") if self.sizes[c]]
        return [self.cluster_card(c, bucket_links) for c in order]

    def group_card(self, b: Bucket, href: str) -> GroupCard:
        return GroupCard(
            name=b.name,
            href=href,
            size=len(b.hits),
            clusters=[ref(c) for c in b.clusters],
            composition=self.composition(np.asarray(b.hits)),
            hits=self.hit_views([int(h) for h in ranked_sample(b, self.probs, 10)]),
        )

    def bucket_page(self, b: Bucket, positions: Positions, href: str) -> None:
        stars = [int(h) for h in ranked_sample(b, self.probs, 8)]
        shown = [int(h) for h in ranked_sample(b, self.probs, 6)]
        tiles: list[TileView] = []
        for i, h in enumerate(shown):
            t = self.tile(h)
            p = self.probs[h]
            tiles.append(
                TileView(
                    number=i + 1,
                    id=self.space.hit_ids[h],
                    image=t.image,
                    overlay=t.overlay,
                    label=t.label,
                    activation=float(self.activation[h]),
                    entropy=float(self.entropy[h]),
                    pattern=self.pattern(h),
                    probabilities=[(ref(c), float(p[c])) for c in np.argsort(-p)[:3]],
                )
            )
        page = render(
            "bucket.html.j2",
            name=b.name,
            clusters=[ref(c) for c in b.clusters],
            size=len(b.hits),
            single=len(b.clusters) == 1,
            threshold=b.threshold,
            composition=self.composition(np.asarray(b.hits)),
            map=embed(self.map_figure(positions, "cluster", stars, shown), "80vh"),
            tiles=tiles,
        )
        (self.out / href).write_text(page)

    def write(
        self, out: Path, buckets: list[Bucket], positions: dict[int, tuple[float, float]]
    ) -> None:
        self.out = out
        (out / "patterns").mkdir(parents=True, exist_ok=True)
        positions = {int(k): (float(v[0]), float(v[1])) for k, v in positions.items()}
        hrefs, used = {}, set()
        for b in buckets:
            name = slug(b.name)
            while name in used:
                name += "-x"
            used.add(name)
            hrefs[id(b)] = f"{name}.html"
            self.bucket_page(b, positions, hrefs[id(b)])
        bucket_links: BucketLinks = {}
        for b in buckets:
            if len(b.clusters) == 1:
                bucket_links.setdefault(b.clusters[0], []).append((b.threshold, hrefs[id(b)]))
        page = render(
            "index.html.j2",
            unit=self.unit,
            stats=self.header_stats(),
            activations=self.activations() if self.all_activations is not None else None,
            hit_ranges=self.hit_ranges(),
            forest=self.forest(),
            membership=self.membership(),
            maps=self.maps(positions),
            cluster_cards=self.cluster_cards(bucket_links),
            group_cards=[self.group_card(b, hrefs[id(b)]) for b in buckets if len(b.clusters) > 1],
            buckets=[{"name": b.name, "href": hrefs[id(b)], "size": len(b.hits)} for b in buckets],
        )
        (self.out / "index.html").write_text(page)
