import argparse
from pathlib import Path

from neural_interp.experiments import compare, mixed4e_55


def main() -> None:
    parser = argparse.ArgumentParser(prog="nint")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "report"):
        p = sub.add_parser(name)
        p.add_argument("--pool", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--n-images", type=int, default=2000)
        p.add_argument("--image-list", type=Path)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--selection", choices=["elbow", "bucketed"], default="elbow")
        p.add_argument("--budget", type=int, default=14000)
    p = sub.add_parser("compare")
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        compare.compare(args.ours, args.reference, args.out)
        return
    cfg = mixed4e_55.Config(
        pool=args.pool, out=args.out, n_images=args.n_images, image_list=args.image_list, seed=args.seed,
        selection=args.selection, budget=args.budget,
    )
    getattr(mixed4e_55, args.command)(cfg)
