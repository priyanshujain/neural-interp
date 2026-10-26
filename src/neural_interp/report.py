import html
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from neural_interp.buckets import Bucket
from neural_interp.space import HitSpace

PALETTE = qualitative.Dark24 + qualitative.Light24
SIZE = 640


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "bucket"


def page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:2px 8px;text-align:right}"
        ".card{display:inline-block;vertical-align:top;margin:6px;padding:6px;border:1px solid #ccc;width:240px}"
        ".card img{max-width:100%}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
    )


def chosen(space: HitSpace, hits: np.ndarray, n: int) -> np.ndarray:
    if len(hits) <= n:
        return hits[np.argsort(-space.activation[hits], kind="stable")]
    ranked = hits[np.argsort(-space.activation[hits], kind="stable")]
    return ranked[np.unique(np.linspace(0, len(ranked) - 1, n).round().astype(int))]


def hover(space: HitSpace, hit: int) -> str:
    return f"{space.hit_ids[hit]}<br>act {space.activation[hit]:.3g}<br>cluster {space.cluster[hit]}"


def figure(
    space: HitSpace,
    positions: dict[int, tuple[float, float]],
    labels: dict[int, str],
    scale: int = 1,
    highlight: np.ndarray | None = None,
    stars: np.ndarray | None = None,
) -> go.Figure:
    nodes = list(positions)
    fig = go.Figure()
    faded = highlight is not None
    names = sorted(set(labels.values()), key=lambda s: (len(s), s))
    for i, name in enumerate(names):
        members = [n for n in nodes if labels[n] == name]
        fig.add_trace(go.Scattergl(
            x=[positions[n][0] for n in members], y=[positions[n][1] for n in members],
            mode="markers", name=name,
            marker=dict(size=5 * scale ** 0.5, color=PALETTE[i % len(PALETTE)], opacity=0.25 if faded else 0.8),
            text=[hover(space, n) for n in members], hoverinfo="text",
        ))
    if faded:
        shown = [int(n) for n in highlight if int(n) in positions]
        fig.add_trace(go.Scattergl(
            x=[positions[n][0] for n in shown], y=[positions[n][1] for n in shown],
            mode="markers", name="bucket",
            marker=dict(size=8 * scale ** 0.5, color="rgba(0,0,0,0)", line=dict(color="black", width=1.5)),
            text=[hover(space, n) for n in shown], hoverinfo="text",
        ))
    if stars is not None:
        shown = [(k + 1, int(n)) for k, n in enumerate(stars) if int(n) in positions]
        fig.add_trace(go.Scatter(
            x=[positions[n][0] for _, n in shown], y=[positions[n][1] for _, n in shown],
            mode="markers+text", name="starred", text=[str(k) for k, _ in shown], textposition="top center",
            marker=dict(symbol="star", size=14 * scale ** 0.5, color="gold", line=dict(color="black", width=1)),
            hovertext=[hover(space, n) for _, n in shown], hoverinfo="text",
        ))
    fig.update_layout(
        width=SIZE * scale, height=SIZE * scale, template="plotly_white",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def write_map(out: Path, name: str, title: str, make: Callable[[int], go.Figure], first: bool = True) -> str:
    big = f"{name}_4x.html"
    (out / big).write_text(page(f"{title} (4x)", make(4).to_html(full_html=False, include_plotlyjs="cdn")))
    return make(1).to_html(full_html=False, include_plotlyjs="cdn" if first else False) + f"<p><a href='{big}'>4x map</a></p>"


def composition(space: HitSpace, group_of: Callable[[int], str] | None) -> str:
    clusters = np.asarray(space.cluster)
    groups = [group_of(h) for h in range(len(clusters))] if group_of else []
    kinds = sorted(set(groups))
    head = "".join(f"<th>{html.escape(k)}</th>" for k in kinds)
    rows = []
    for c in range(int(np.asarray(space.probs).shape[1])):
        members = np.flatnonzero(clusters == c)
        tally = Counter(groups[m] for m in members) if groups else Counter()
        cells = "".join(
            f"<td>{tally[k]} ({100 * tally[k] / len(members):.0f}%)</td>" if len(members) else "<td>0</td>"
            for k in kinds
        )
        mean = f"{space.activation[members].mean():.3g}" if len(members) else "-"
        rows.append(f"<tr><td>{c}</td><td>{len(members)}</td>{cells}<td>{mean}</td></tr>")
    return (
        f"<table><tr><th>cluster</th><th>size</th>{head}<th>mean activation</th></tr>"
        + "".join(rows) + "</table>"
    )


def card_html(space: HitSpace, number: int, hit: int, body: str) -> str:
    p = np.asarray(space.probs[hit])
    top = np.argsort(-p, kind="stable")[:3]
    probs = ", ".join(f"c{c} {p[c]:.2f}" for c in top if p[c] > 0)
    return (
        f"<div class='card'><b>#{number}</b> {html.escape(space.hit_ids[hit])}<br>"
        f"act {space.activation[hit]:.3g} | entropy {space.entropy[hit]:.2f}<br>{probs}<br>{body}</div>"
    )


def write_report(
    out: Path,
    space: HitSpace,
    buckets: list[Bucket],
    positions: dict[int, tuple[float, float]],
    card: Callable[[int], str],
    group_of: Callable[[int], str] | None = None,
    starred: int = 12,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    positions = {int(k): (float(v[0]), float(v[1])) for k, v in positions.items()}
    by_cluster = {n: f"cluster {space.cluster[n]}" for n in positions}
    by_group = {n: group_of(n) for n in positions} if group_of else None
    cards: dict[int, str] = {}

    links, used = [], set()
    for b in buckets:
        name = slug(b.name)
        while name in used:
            name += "-x"
        used.add(name)
        hits = np.asarray(b.hits, dtype=int)
        drawn = np.array([h for h in hits if h in positions], dtype=int)
        stars = chosen(space, drawn if len(drawn) else hits, starred)
        for h in stars:
            if int(h) not in cards:
                cards[int(h)] = card(int(h))
        body = write_map(out, name, b.name, lambda s: figure(space, positions, by_cluster, s, hits, stars))
        body += f"<p>{len(hits)} hits, {len(drawn)} on map, threshold {b.threshold}</p>"
        body += "".join(card_html(space, k + 1, int(h), cards[int(h)]) for k, h in enumerate(stars))
        (out / f"{name}.html").write_text(page(b.name, body))
        links.append(f"<tr><td style='text-align:left'><a href='{name}.html'>{html.escape(b.name)}</a></td>"
                      f"<td>{len(hits)}</td></tr>")

    body = "<h2>Buckets</h2><table><tr><th>bucket</th><th>size</th></tr>" + "".join(links) + "</table>"
    body += "<h2>Map by cluster</h2>" + write_map(out, "map_cluster", "map by cluster",
                                                  lambda s: figure(space, positions, by_cluster, s))
    if by_group:
        body += "<h2>Map by group</h2>" + write_map(out, "map_group", "map by group",
                                                    lambda s: figure(space, positions, by_group, s), False)
    body += "<h2>Cluster composition</h2>" + composition(space, group_of)
    (out / "index.html").write_text(page("Hit space report", body))
