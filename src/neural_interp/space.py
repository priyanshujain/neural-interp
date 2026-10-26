import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_interp.density_peaks import Forest
from neural_interp.membership import Membership


@dataclass
class HitSpace:
    hit_ids: list[str]
    activation: np.ndarray
    rows: np.ndarray
    point: np.ndarray
    counts: np.ndarray
    dist: np.ndarray
    forest: Forest
    membership: Membership

    @property
    def cluster(self) -> np.ndarray:
        return self.forest.tree[self.point]

    @property
    def probs(self) -> np.ndarray:
        return self.membership.probs[self.point]

    @property
    def entropy(self) -> np.ndarray:
        return self.membership.entropy[self.point]

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "distances.npy", self.dist)
        f, m = self.forest, self.membership
        np.savez_compressed(
            out / "space.npz",
            activation=self.activation, rows=self.rows, point=self.point, counts=self.counts,
            rho=f.rho, delta=f.delta, parent=f.parent, roots=f.roots, tree=f.tree, depth=f.depth,
            geodesic=m.geodesic, probs=m.probs, entropy=m.entropy,
            peak_members=np.concatenate(m.peaks), peak_sizes=np.array([len(p) for p in m.peaks]),
        )
        meta = {
            "hit_ids": self.hit_ids,
            "rho_cut": f.rho_cut, "delta_cut": f.delta_cut,
            "temperature": m.temperature, "touching": m.touching,
        }
        (out / "space.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, out: Path) -> "HitSpace":
        z = np.load(out / "space.npz")
        meta = json.loads((out / "space.json").read_text())
        forest = Forest(z["rho"], z["delta"], z["parent"], z["roots"], z["tree"], z["depth"],
                        meta["rho_cut"], meta["delta_cut"])
        peaks = np.split(z["peak_members"], np.cumsum(z["peak_sizes"])[:-1])
        membership = Membership(peaks, z["geodesic"], z["probs"], z["entropy"], meta["temperature"],
                                [tuple(p) for p in meta["touching"]])
        return cls(meta["hit_ids"], z["activation"], z["rows"], z["point"], z["counts"],
                   np.load(out / "distances.npy", mmap_mode="r"), forest, membership)


def summary(space: HitSpace) -> dict:
    return {
        "hits": len(space.hit_ids),
        "points": len(space.counts),
        "clusters": len(space.forest.roots),
        "confident": int((space.entropy < 0.5).sum()),
        "cluster_sizes": np.bincount(space.cluster).tolist(),
        "rho_cut": space.forest.rho_cut,
        "delta_cut": space.forest.delta_cut,
    }
