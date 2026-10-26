from importlib.resources import files
from typing import Any

import jinja2
import plotly.graph_objects as go
from markupsafe import Markup
from plotly.colors import qualitative


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


def color(cluster: int) -> str:
    return PALETTE[int(cluster) % len(PALETTE)]


def static_text(name: str) -> str:
    return files("neural_interp.report").joinpath("static", name).read_text(encoding="utf-8")


def format_value(value: Any, spec: str) -> str:
    return format(value, spec)


environment = jinja2.Environment(
    loader=jinja2.PackageLoader("neural_interp.report", "templates"),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.StrictUndefined,
    extensions=["jinja2.ext.do"],
)
environment.filters["fmt"] = format_value
page_assets = {"css": Markup(static_text("report.css")), "js": Markup(static_text("report.js"))}


def render(template: str, **context: Any) -> str:
    return environment.get_template(template).render(**page_assets, **context)


def embed(fig: go.Figure, height: str = "70vh", div_id: str | None = None) -> Markup:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        autosize=True,
        margin=dict(l=60, r=20, t=30, b=40),
        font=dict(size=13),
    )
    return Markup(
        fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            default_width="100%",
            default_height=height,
            config={"responsive": True, "displaylogo": False},
            div_id=div_id,
        )
    )
