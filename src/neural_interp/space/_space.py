import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_interp.cluster import DensityPeakForest, SoftMembership
from neural_interp.typing import FloatArray, IntArray


@dataclass
class HitSpace:
    hit_ids: list[str]
    activation: FloatArray
    rows: FloatArray
    point: IntArray
    counts: IntArray
    distances: FloatArray
    forest: DensityPeakForest
    membership: SoftMembership

    @property
    def probabilities(self) -> FloatArray:
        return self.membership.probabilities_[self.point]

    @property
    def cluster(self) -> IntArray:
        return self.membership.labels_[self.point]

    @property
    def tree(self) -> IntArray:
        return self.forest.labels_[self.point]

    @property
    def entropy(self) -> FloatArray:
        return self.membership.entropy_[self.point]

    def summary(self) -> dict[str, Any]:
        return {
            "hits": len(self.hit_ids),
            "points": len(self.counts),
            "clusters": len(self.forest.roots_),
            "cluster_differs_from_tree": int((self.cluster != self.tree).sum()),
            "confident": int((self.entropy < 0.5).sum()),
            "rho_cut": self.forest.rho_cut_,
            "delta_cut": self.forest.delta_cut_,
            "cluster_sizes": np.bincount(self.cluster).tolist(),
        }

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "distances.npy", self.distances)
        f, m = self.forest, self.membership
        np.savez_compressed(
            out / "space.npz",
            activation=self.activation,
            rows=self.rows,
            point=self.point,
            counts=self.counts,
            rho=f.rho_,
            delta=f.delta_,
            parent=f.parent_,
            roots=f.roots_,
            tree=f.labels_,
            depth=f.depth_,
            geodesic=m.geodesic_,
            probabilities=m.probabilities_,
            entropy=m.entropy_,
            labels=m.labels_,
            peak_members=np.concatenate(m.peaks_),
            peak_sizes=np.array([len(p) for p in m.peaks_]),
        )
        meta = {
            "hit_ids": self.hit_ids,
            "n_neighbors": f.n_neighbors,
            "peak_share": m.peak_share,
            "rho_cut": f.rho_cut_,
            "delta_cut": f.delta_cut_,
            "temperature": m.temperature_,
            "touching": m.touching_,
        }
        (out / "space.json").write_text(json.dumps(meta))
        (out / "summary.json").write_text(json.dumps(self.summary(), indent=1))

    @classmethod
    def load(cls, out: Path) -> "HitSpace":
        z = np.load(out / "space.npz")
        meta = json.loads((out / "space.json").read_text())
        forest = DensityPeakForest(meta["n_neighbors"])
        forest.rho_, forest.delta_, forest.parent_ = z["rho"], z["delta"], z["parent"]
        forest.roots_, forest.labels_, forest.depth_ = z["roots"], z["tree"], z["depth"]
        forest.rho_cut_, forest.delta_cut_ = meta["rho_cut"], meta["delta_cut"]
        membership = SoftMembership(meta["n_neighbors"], meta["peak_share"])
        membership.peaks_ = list(np.split(z["peak_members"], np.cumsum(z["peak_sizes"])[:-1]))
        membership.geodesic_, membership.probabilities_ = z["geodesic"], z["probabilities"]
        membership.entropy_, membership.labels_ = z["entropy"], z["labels"]
        membership.temperature_ = meta["temperature"]
        membership.touching_ = [(int(a), int(b)) for a, b in meta["touching"]]
        return cls(
            meta["hit_ids"],
            z["activation"],
            z["rows"],
            z["point"],
            z["counts"],
            np.load(out / "distances.npy", mmap_mode="r"),
            forest,
            membership,
        )
