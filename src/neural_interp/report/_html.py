import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.colors import qualitative

from neural_interp.buckets import Bucket, ranked_sample
from neural_interp.cluster import knn_graph
from neural_interp.selection import elbow_index
from neural_interp.space import HitSpace
from neural_interp.typing import FloatArray

Positions = dict[int, tuple[float, float]]
BucketLinks = dict[int, list[tuple[str, str]]]


def bright(hex_color: str) -> bool:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.25 < 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.85


PALETTE = [
    c
    for c in qualitative.Plotly
    + qualitative.D3
    + qualitative.Light24
    + qualitative.Alphabet
    + qualitative.Dark24
    if bright(c)
]
PANEL = "#1f2229"
CDN = "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#16181d;color:#d7dae0;\
font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
main{max-width:1800px;margin:0 auto;padding:24px 32px 80px}
a{color:#8ab4ff;text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:28px;margin:0 0 4px}h2{font-size:22px;margin:48px 0 4px;padding-top:16px;\
border-top:1px solid #2c3039}
h2 .step{color:#7b8191;font-weight:400;margin-right:8px}
.lede{color:#9aa1ae;max-width:1100px;margin:4px 0 16px}.lede b{color:#fff}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.stat{background:#1f2229;border-radius:8px;padding:10px 16px}.stat b{display:block;font-size:22px;\
color:#fff}
.stat span{color:#9aa1ae;font-size:13px}
details.guide{background:#1f2229;border-radius:8px;padding:12px 18px;margin:12px 0;max-width:1200px}
details summary{cursor:pointer;color:#c9ccd3;font-weight:600}
.guide dt{font-weight:600;color:#fff;margin-top:10px}.guide dd{margin:2px 0 0 0;color:#aab0bc}
.plot{width:100%;background:#1f2229;border-radius:8px;margin:8px 0}
.tabs button{background:#2a2e37;color:#d7dae0;border:0;padding:8px 16px;border-radius:6px 6px 0 0;\
cursor:pointer;font-size:14px}
.tabs button.on{background:#3b4150;color:#fff}
.tabpane{display:none}.tabpane.on{display:block}
.card{background:#1f2229;border-radius:10px;margin:28px 0;overflow:hidden;\
border-left:6px solid var(--c)}
.card header{background:var(--c);color:#fff;padding:12px 20px;display:flex;flex-wrap:wrap;\
align-items:center;gap:16px}
.card header h3{margin:0;font-size:22px}.card header span{opacity:.92}
.card header .legend{color:#fff}
.card .body{padding:16px 20px}
.strip{display:grid;grid-template-columns:repeat(10,1fr);gap:4px;margin:4px 0}
.strip img{width:100%;aspect-ratio:1;object-fit:cover;display:block;border-radius:3px;\
background:#000}
.label{font-size:12px;color:#9aa1ae;margin:8px 0 2px}
.meta{display:flex;flex-wrap:wrap;align-items:center;gap:24px;font-size:14px;color:#aab0bc;\
margin:4px 0 10px}
.meta b{color:#fff}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;width:260px;background:#333}
.legend{display:flex;gap:12px;font-size:12px;color:#aab0bc}.legend i{display:inline-block;\
width:10px;height:10px;border-radius:2px;margin-right:4px}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff !important;font-size:13px;\
margin-right:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:16px;\
margin-top:16px}
.tile{background:#262a32;border-radius:8px;padding:10px}
.tile .pair{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:6px}
.tile .pair img{width:100%;border-radius:3px;display:block}
.tile .pattern{display:block;height:88px;margin:6px auto 0;image-rendering:pixelated;\
border-radius:3px}
.tile .num{font-weight:700;color:#ffd600}
.tile .id{font-size:12px;color:#7b8191;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.probs{margin-top:6px}.probs div{display:flex;align-items:center;gap:6px;font-size:12px}
.probs .fill{height:8px;border-radius:4px}
table{border-collapse:collapse;font-size:14px}td,th{padding:4px 12px;\
border-bottom:1px solid #2c3039;text-align:left}
th{color:#9aa1ae;font-weight:500}
details.patterns summary{font-weight:500;margin-top:8px}
"""

TABS_JS = """<script>
function tab(g,n){document.querySelectorAll('[data-g="'+g+'"]').forEach(\
e=>e.classList.toggle('on',e.dataset.n===n));
window.dispatchEvent(new Event('resize'));}
</script>"""


@dataclass
class Tile:
    image: str
    overlay: str
    label: str


def color(cluster: int) -> str:
    return PALETTE[int(cluster) % len(PALETTE)]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "bucket"


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style>{CDN}{TABS_JS}</head>"
        f"<body><main>{body}</main></body></html>"
    )


def embed(fig: go.Figure, height: str = "70vh", div_id: str | None = None) -> str:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        autosize=True,
        margin=dict(l=60, r=20, t=30, b=40),
        font=dict(size=13),
    )
    inner = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        default_width="100%",
        default_height=height,
        config={"responsive": True, "displaylogo": False},
        div_id=div_id,
    )
    return f"<div class='plot'>{inner}</div>"


def section(step: str, title: str, lede: str) -> str:
    return (
        f"<h2><span class='step'>{step}</span>{html.escape(title)}</h2><p class='lede'>{lede}</p>"
    )


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

    def chip(self, c: int, prefix: str = "") -> str:
        return f"<a class='chip' href='{prefix}#cluster-{c}' style='background:{color(c)}'>c{c}</a>"

    def tabs(self, group: str, panes: dict[str, str]) -> str:
        names = list(panes)
        buttons = "".join(
            f"<button data-g='{group}' data-n='{i}' class='{'on' if i == 0 else ''}' "
            f"onclick=\"tab('{group}','{i}')\">{html.escape(n)}</button>"
            for i, n in enumerate(names)
        )
        bodies = "".join(
            f"<div data-g='{group}' data-n='{i}' class='tabpane "
            f"{'on' if i == 0 else ''}'>{panes[n]}</div>"
            for i, n in enumerate(names)
        )
        return f"<div class='tabs'>{buttons}</div>{bodies}"

    def guide(self) -> str:
        items = [
            (
                "Hit",
                "A position where the unit fires strongly: an image patch for a vision unit, a "
                "token for a text unit. Hits are the positive activations past the elbow of the "
                "sorted curve (section 1).",
            ),
            (
                "Hit vector",
                "The exact contribution of every upstream channel to one hit: upstream "
                "activation times the unit's weight for that channel. It sums to the unit's "
                "pre-activation minus its bias. Drawn as a grid of channels: green pushes the "
                "unit up, red pushes it down.",
            ),
            (
                "Distance",
                "Weighted Jaccard between hit vectors, positive and negative parts compared "
                "separately. 0 means the same channels drove both hits in the same amounts, 1 "
                "means no channel in common.",
            ),
            (
                "Forest and clusters",
                "Each hit links to its nearest denser hit. Hits that are both dense and far "
                "from anything denser become roots, and every other hit climbs to one root. "
                "Each root's tree is a cluster (section 3).",
            ),
            (
                "Probability and entropy",
                "Each hit gets a probability for every cluster from its graph distance to that "
                "cluster's peak. A hit's cluster is its most probable one. Entropy in bits says "
                "how split it is; under 0.5 is confident.",
            ),
            (
                "Map",
                "Hits laid out by a spring layout on their real distances, with an extra pull "
                "toward the peaks of the clusters they likely belong to. Nothing is projected: "
                "a hit between two clusters sits between their peaks.",
            ),
            (
                "DeepLift",
                "Which input pixels drove this hit, against a black baseline. Yellow raises the "
                "unit, blue lowers it. The box marks the hit's position.",
            ),
            (
                "Group clusters",
                "Hits that need the same set of clusters to cover 85% of their probability: the "
                "borders between clusters.",
            ),
        ]
        body = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items)
        return (
            "<details class='guide'><summary>How to read this "
            f"report</summary><dl>{body}</dl></details>"
        )

    def header(self) -> str:
        stats = [
            (
                f"{len(self.all_activations):,}" if self.all_activations is not None else "-",
                "activations",
            ),
            (f"{len(self.space.hit_ids):,}", "hits"),
            (f"{self.n_clusters}", "clusters"),
            (f"{int((self.entropy < 0.5).sum()):,}", "confident hits"),
            (f"{len(self.space.membership.touching_)}", "touching cluster pairs"),
        ]
        cells = "".join(f"<div class='stat'><b>{v}</b><span>{k}</span></div>" for v, k in stats)
        return f"<h1>{html.escape(self.unit)}</h1><div class='stats'>{cells}</div>{self.guide()}"

    def activations(self) -> str:
        values = self.all_activations
        assert values is not None
        cut = float(self.activation.min())
        positive = np.sort(values[values > 0])
        n = len(positive)
        elbow = n - 1 - elbow_index(positive[::-1])
        chord = positive[0] + (positive[-1] - positive[0]) * elbow / (n - 1)
        out = section(
            "1",
            "Activations and the elbow",
            f"All {len(values):,} activations of the unit, of which {n:,} are positive. Sorted "
            "from low to high, most sit in a long flat stretch "
            "and a few thousand shoot up at the end. The elbow is where the curve bends: draw a "
            "straight line from the first point to the last, "
            "and take the point where the curve is furthest below it. Everything right of the "
            f"elbow is a hit: activation &ge; <b>{cut:.4g}</b>, "
            f"<b>{len(self.activation):,}</b> hits.",
        )
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
        hist_note = (
            "<p class='lede'>How many activations take each value. Bars are counts on a log "
            "scale, so each gridline is ten times the one below. "
            "Values at or below zero are the unit not firing; the pile just above zero is weak "
            "firing; the thin right tail, shaded, is the hits.</p>"
        )
        return out + self.tabs(
            "act",
            {"Sorted curve": embed(curve, "65vh"), "Histogram": hist_note + embed(hist, "60vh")},
        )

    def hit_ranges(self) -> str:
        out = section(
            "2",
            "Activation range of each cluster",
            "Each box is a cluster, largest first. Clusters are not activation bands: strong "
            "and moderate hits share a cluster when the same channels drive them.",
        )
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
        return out + embed(fig, "55vh")

    def peak_separation(self) -> str:
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
        return (
            "<p class='lede'>How clearly each peak stands out. Every root is dense enough; its "
            "bar is how far it is from anything denser. "
            "Tall bars are clear, separate peaks; bars just above the cut are marginal and would "
            "merge with a slightly stricter cut. "
            "Grey bars are the best dense points that did not make it.</p>" + embed(fig, "60vh")
        )

    def forest_up_close(self, per_tree: int = 100, seed: int = 0) -> str:
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
        pos = nx.spring_layout(g, weight="weight", iterations=200, seed=seed)
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
        chips = "".join(
            f"<button class='pick{' on' if t in default else ''}' data-t='{t}' "
            f"style='--c:{color(t)}' onclick='pick(this)'>c{t}</button>"
            for t in order
        )
        script = f"""<script>
function upclose(){{const on=[...document.querySelectorAll('.pick.on')].map(b=>+b.dataset.t);
const vis=[];for(let t=0;t<{n_trees};t++){{const v=on.includes(t);vis.push(v,v,v);}}
Plotly.restyle('upclose',{{visible:vis}});\
Plotly.relayout('upclose',{{'xaxis.autorange':true,'yaxis.autorange':true}});}}
function pick(b){{b.classList.toggle('on');upclose();}}
function pickAll(v){{document.querySelectorAll('.pick').forEach(b=>b.classList.toggle('on',v));\
upclose();}}
</script>"""
        style = (
            "<style>.pick{background:#2a2e37;color:#9aa1ae;border:2px solid "
            "var(--c);border-radius:12px;padding:2px 10px;margin:2px;cursor:pointer}"
            ".pick.on{background:var(--c);color:#fff}.picker{margin:6px 0 4px}.picker "
            ".ctl{background:#3b4150;color:#fff;border:0;border-radius:6px;"
            "padding:3px 10px;margin-right:6px;cursor:pointer}</style>"
        )
        return (
            f"<p class='lede'>A slice of the real forest: up to {per_tree} random hits from every "
            "cluster, plus every hit on their path up to the root, "
            "laid out once on real distances, so near means similar and positions stay fixed as "
            "you add clusters. Each line joins a hit to its parent, "
            "the nearest denser hit; bigger dots are denser; stars are roots. Pick clusters "
            "below to compare them; it starts with the two large clusters that touch most.</p>"
            + style
            + script
            + (
                "<div class='picker'><button class='ctl' onclick='pickAll(true)'>all</button>"
                f"<button class='ctl' onclick='pickAll(false)'>none</button>{chips}</div>"
            )
            + embed(fig, "80vh", div_id="upclose")
        )

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

    def forest(self) -> str:
        f = self.space.forest
        counts = np.asarray(self.space.counts)
        point_cluster = np.asarray(self.space.membership.probabilities_).argmax(1)
        n_roots = len(f.roots_)
        out = section(
            "3",
            "Density-peak forest",
            "Every point has a density rho (from the distance to its 5th neighbour) and a delta "
            "(distance to the nearest denser point). "
            f"Roots are the points above both elbow cuts, rho &ge; <b>{f.rho_cut_:.3f}</b> and "
            f"delta &ge; <b>{f.delta_cut_:.3f}</b>: "
            f"dense, and far from anything denser. This gives <b>{n_roots}</b> roots.",
        )
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
        return out + self.tabs(
            "forest",
            {
                "Decision graph": "<p class='lede'>Each dot is a hit, coloured by the tree "
                "it ends up in. Right means a crowded neighbourhood; up means it must travel far "
                "to reach anything denser. "
                "Most hits sit low: a denser hit is close by, so they are on the slope of some "
                "hill. Hilltops are both crowded and far from anything denser, "
                "so they land in the top-right corner, past both cut lines. High dots on the "
                "left are loners, not centres.</p>" + embed(dg, "70vh"),
                "Peak separation": self.peak_separation(),
                "Forest up close": self.forest_up_close(),
                "Forest landscape": "<p class='lede'>Each tree drawn as a hill: height is density, "
                "lines are parent links, the root sits on top. "
                "The horizontal position is only the order of points along their tree, not a "
                "distance; read this chart for depth and density. "
                "Click a cluster in the legend to isolate it, double-click others to add them.</p>"
                + embed(land, "75vh"),
            },
        )

    def membership(self) -> str:
        moved = int((self.cluster != np.asarray(self.space.tree)).sum())
        out = section(
            "4",
            "Soft membership",
            f"Probabilities come from graph distance to each cluster's peak. <b>{moved:,}</b> "
            "hits are most probable in a different "
            "cluster than the tree they grew in; from here on a hit's cluster is its most "
            "probable one. Hits under 0.5 bits of entropy are confident.",
        )
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
        return out + self.tabs(
            "member",
            {
                "Entropy": embed(hist, "45vh"),
                "Which clusters touch": "<p class='lede'>Row = cluster, column = neighbour. "
                "Each cell is the share of the cluster's mutual-neighbour edges that land in the "
                "neighbour. "
                "The diagonal is how self-contained a cluster is.</p>" + embed(heat, "80vh"),
            },
        )

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

    def maps(self, positions: Positions) -> str:
        out = section(
            "5",
            "Map of the hit space",
            "Hits sampled from every bucket plus each cluster's root, laid out on their real "
            "distances with a pull toward likely clusters' peaks. "
            "Click a legend entry to isolate it. Switch to <b>By class</b> to see which inputs "
            "each region comes from.",
        )
        panes = {"By cluster": embed(self.map_figure(positions, "cluster"), "85vh")}
        if self.groups is not None:
            panes["By class"] = embed(self.map_figure(positions, "group"), "85vh")
        return out + self.tabs("map", panes)

    def composition_bar(self, members: np.ndarray) -> str:
        if self.groups is None or not len(members):
            return ""
        names, counts = np.unique(self.groups[members], return_counts=True)
        share = counts / counts.sum()
        segments = "".join(
            f"<div title='{n} {s:.0%}' "
            f"style='width:{s * 100:.1f}%;background:{self.group_color(n)}'></div>"
            for n, s in zip(names, share)
        )
        legend = "".join(
            f"<span><i style='background:{self.group_color(n)}'></i>{n} {s:.0%}</span>"
            for n, s in sorted(zip(names, share), key=lambda t: -t[1])
        )
        return f"<div><div class='bar'>{segments}</div><div class='legend'>{legend}</div></div>"

    def strips(self, hits: list[int], caption: str) -> str:
        tiles = [self.tile(h) for h in hits]
        ids = [html.escape(self.space.hit_ids[h]) for h in hits]
        images = "".join(
            f"<img loading='lazy' src='{t.image}' title='{i}'>" for t, i in zip(tiles, ids)
        )
        overlays = "".join(
            f"<img loading='lazy' src='{t.overlay}' title='{html.escape(t.label)} | act "
            f"{self.activation[h]:.4g}'>"
            for t, h in zip(tiles, hits)
        )
        patterns = "".join(
            f"<img loading='lazy' src='{self.pattern(h)}' style='image-rendering:pixelated' "
            f"title='{i}'>"
            for h, i in zip(hits, ids)
        )
        return (
            f"<div class='label'>{caption}</div><div class='strip'>{images}</div>"
            "<div class='label'>DeepLift: which pixels drove each hit (yellow up, blue "
            f"down)</div><div class='strip'>{overlays}</div>"
            "<details class='patterns'><summary>Hit vectors: contribution of each upstream "
            "channel</summary>"
            f"<div class='strip'>{patterns}</div></details>"
        )

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

    def cluster_card(self, c: int, bucket_links: BucketLinks) -> str:
        members = np.flatnonzero(self.cluster == c)
        sources = len({source_of(self.space.hit_ids[h]) for h in members})
        acts = self.activation[members]
        confident = (self.entropy[members] < 0.5).mean()
        inside = self.contacts[c, c] / max(self.contacts[c].sum(), 1)
        touches = "".join(
            f"{self.chip(k)}<span style='margin-right:8px'>{s:.0%}</span>"
            for k, s in self.neighbours(c)
        )
        links = " ".join(f"<a href='{href}'>{label}</a>" for label, href in bucket_links.get(c, []))
        meta = (
            f"<div class='meta'><span>activation <b>{acts.min():.3g} to {acts.max():.3g}</b>, "
            f"median {np.median(acts):.3g}</span>"
            f"<span>confident <b>{confident:.0%}</b></span>"
            f"<span>self-contained <b>{inside:.0%}</b></span>"
            f"<span>nearest clusters {touches or 'none'}</span>"
            f"<span>mean hit vector <img src='{self.cluster_pattern(c)}' "
            "style='height:66px;vertical-align:middle;image-rendering:pixelated;border-radius:3px"
            "'></span>"
            f"<span>bucket pages {links or 'none'}</span></div>"
        )
        return (
            f"<section class='card' id='cluster-{c}' style='--c:{color(c)}'><header><h3>Cluster "
            f"{c}</h3>"
            f"<span>n={len(members):,} &middot; {sources:,} distinct "
            f"inputs</span>{self.composition_bar(members)}</header>"
            "<div "
            f"class='body'>{meta}"
            f"{self.strips(self.core_hits(c), 'Most confident hits, one per input')}</div></sectio"
            "n>"
        )

    def clusters(self, bucket_links: BucketLinks) -> str:
        out = section(
            "6",
            "Clusters",
            "One card per cluster, largest first. The rows show the hits most confidently in the "
            "cluster, one per input, "
            "their DeepLift maps, and their hit vectors. The bar in the header shows which "
            "classes the cluster's hits come from.",
        )
        order = [int(c) for c in np.argsort(-self.sizes, kind="stable") if self.sizes[c]]
        return out + "".join(self.cluster_card(c, bucket_links) for c in order)

    def group_card(self, b: Bucket, href: str) -> str:
        hits = [int(h) for h in ranked_sample(b, self.probs, 10)]
        gradient = ", ".join(color(c) for c in b.clusters)
        return (
            f"<section class='card' style='--c:{color(b.clusters[0])}'><header "
            f"style='background:linear-gradient(90deg,{gradient})'>"
            f"<h3>{html.escape(b.name)}</h3><span>n={len(b.hits):,}</span>"
            f"{self.composition_bar(np.asarray(b.hits))}"
            f"<a href='{href}' style='color:#fff;margin-left:auto'>open page &rarr;</a></header>"
            "<div class='body'><div class='meta'><span>clusters "
            f"{''.join(self.chip(c) for c in b.clusters)}</span></div>"
            f"{
                self.strips(
                    hits, 'Hits ordered from the most to the least probable in the leading cluster'
                )
            }</div></section>"
        )

    def groups_section(self, sets: list[tuple[Bucket, str]]) -> str:
        out = section(
            "7",
            "Group clusters",
            "Hits that need the same set of clusters to cover 85% of their probability: the "
            "borders between clusters. "
            "Each row runs from hits leaning to the leading cluster to hits leaning away from it.",
        )
        return out + ("".join(self.group_card(b, href) for b, href in sets) or "<p>none</p>")

    def bucket_page(self, b: Bucket, positions: Positions, href: str) -> None:
        stars = [int(h) for h in ranked_sample(b, self.probs, 8)]
        shown = [int(h) for h in ranked_sample(b, self.probs, 6)]
        tiles = []
        for i, h in enumerate(shown):
            t = self.tile(h)
            p = self.probs[h]
            bars = "".join(
                f"<div><span style='width:34px'>c{c}</span><div class='fill' "
                f"style='width:{p[c] * 160:.0f}px;background:{color(c)}'></div>{p[c]:.2f}</div>"
                for c in np.argsort(-p)[:3]
            )
            tiles.append(
                f"<div class='tile'><div><span class='num'>#{i + 1}</span> {html.escape(t.label)} "
                f"&middot; act {self.activation[h]:.4g} &middot; entropy "
                f"{self.entropy[h]:.2f}</div>"
                "<div class='id' "
                f"title='{html.escape(self.space.hit_ids[h])}'>"
                f"{html.escape(self.space.hit_ids[h])}</div>"
                f"<div class='pair'><img src='{t.image}' title='input'><img src='{t.overlay}' "
                "title='DeepLift'></div>"
                f"<img class='pattern' src='{self.pattern(h)}' title='hit vector'><div "
                f"class='probs'>{bars}</div></div>"
            )
        rule = (
            f"probability &ge; {b.threshold}"
            if len(b.clusters) == 1
            else f"these clusters cover {b.threshold:.0%} of probability"
        )
        body = (
            f"<p><a href='index.html'>&larr; back to report</a></p><h1>{html.escape(b.name)}</h1>"
            "<div class='meta'><span>clusters "
            f"{''.join(self.chip(c, 'index.html') for c in b.clusters)}</span>"
            f"<span><b>{len(b.hits):,}</b> hits, "
            f"{rule}</span>{self.composition_bar(np.asarray(b.hits))}</div>"
            "<p class='lede'>Stars: 8 hits sampled at evenly spaced ranks of the leading "
            "cluster's probability. "
            "The 6 numbered ones are shown below with input, DeepLift, hit vector and top "
            "cluster probabilities.</p>"
            + embed(self.map_figure(positions, "cluster", stars, shown), "80vh")
            + f"<div class='grid'>{''.join(tiles)}</div>"
        )
        (self.out / href).write_text(page(b.name, body))

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
        bucket_links: dict[int, list[tuple[str, str]]] = {}
        for b in buckets:
            if len(b.clusters) == 1:
                bucket_links.setdefault(b.clusters[0], []).append(
                    (f"&ge;{b.threshold}", hrefs[id(b)])
                )
        sets = [(b, hrefs[id(b)]) for b in buckets if len(b.clusters) > 1]
        table = "".join(
            "<tr><td><a "
            f"href='{hrefs[id(b)]}'>{html.escape(b.name)}</a></td><td>{len(b.hits):,}</td></tr>"
            for b in buckets
        )
        body = (
            self.header()
            + (self.activations() if self.all_activations is not None else "")
            + self.hit_ranges()
            + self.forest()
            + self.membership()
            + self.maps(positions)
            + self.clusters(bucket_links)
            + self.groups_section(sets)
            + section(
                "8", "All buckets", "Every bucket with at least 10 hits, each with its own page."
            )
            + f"<table><tr><th>bucket</th><th>hits</th></tr>{table}</table>"
        )
        (self.out / "index.html").write_text(page(f"{self.unit} report", body))
