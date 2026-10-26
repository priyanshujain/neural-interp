import argparse
import importlib

EXPERIMENTS = {
    "embed": ("W_E rows, token ids and token features", ["out"], {}),
    "freq": ("WikiText-103 token frequencies", ["out"], {}),
    "forest": ("GPU density-peak forest on token rows", ["out"], {"mode": ["real", "null"]}),
    "worlds-report": ("HTML report for the token-row forest", ["out", "html_path"], {}),
    "occurrences": ("WikiText occurrences through W_E", ["out"], {}),
    "dims": ("cluster the columns of W_E", ["out"], {}),
    "single-dim": ("one dimension's elbow hits", ["dim", "sign", "out"], {"sign": ["pos", "neg"]}),
    "render": ("HTML report from a saved JSON", ["template", "data_path", "html_path"], {"template": ["dims", "occ", "single"]}),
}


def render(template, data_path, html_path):
    from pathlib import Path

    from neural_interp.lib.report import render as write

    write(template, Path(data_path).read_text(), html_path)


def main():
    parser = argparse.ArgumentParser(prog="nint")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, args, choices) in EXPERIMENTS.items():
        sub = commands.add_parser(name, help=help_text)
        for arg in args:
            sub.add_argument(arg, choices=choices.get(arg))
    args = vars(parser.parse_args())
    name = args.pop("command")
    if name == "render":
        return render(**args)
    importlib.import_module(f"neural_interp.experiments.{name.replace('-', '_')}").run(**args)


if __name__ == "__main__":
    main()
