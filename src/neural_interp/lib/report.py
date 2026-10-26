import json
from pathlib import Path

TEMPLATES = Path(__file__).parent.parent / "templates"


def render(template, data, html_path):
    from plotly.offline import get_plotlyjs

    payload = (data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)).replace("</", "<\\/")
    html = (TEMPLATES / f"{template}.html").read_text().replace("/*PLOTLY*/", get_plotlyjs()).replace("/*DATA*/", payload)
    html_path = Path(html_path)
    html_path.write_text(html, encoding="utf-8")
    print("wrote", html_path, f"{html_path.stat().st_size / 1e6:.1f} MB")


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False))
    print("saved", path)
