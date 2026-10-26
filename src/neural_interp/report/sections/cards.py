from typing import Any

import numpy as np

from neural_interp.buckets import ranked_sample
from neural_interp.report._context import LinkedBucket, ReportContext, ref, source_of
from neural_interp.report._section import Section
from neural_interp.report._views import ClusterCard, GroupCard, GuideEntry


def core_hits(context: ReportContext, cluster: int, n: int = 10) -> list[int]:
    members = np.flatnonzero(context.clusters == cluster)
    members = members[np.argsort(-context.probabilities[members, cluster], kind="stable")]
    picked: list[int] = []
    seen: set[str] = set()
    for h in members:
        key = source_of(context.space.hit_ids[h])
        if key not in seen:
            seen.add(key)
            picked.append(int(h))
            if len(picked) == n:
                break
    return picked


def neighbours(context: ReportContext, cluster: int, n: int = 5) -> list[tuple[int, float]]:
    row = context.contacts[cluster].copy()
    row[cluster] = 0
    total = row.sum()
    if not total:
        return []
    return [(int(k), float(row[k] / total)) for k in np.argsort(-row)[:n] if row[k] > 0]


class ClusterCards(Section):
    title = "Clusters"
    guide = (
        GuideEntry(
            "Hit vector",
            "The exact contribution of every upstream input to one hit: input activation times the"
            " unit's weight for it, summing to the pre-activation minus the bias. Drawn as a grid:"
            " green pushes the unit up, red pushes it down.",
        ),
    )
    template = "sections/cluster_cards.html.j2"

    def shown(self, context: ReportContext) -> list[int]:
        return [c for c in context.clusters_by_size() if context.sizes[c]]

    def hits(self, context: ReportContext) -> set[int]:
        return {h for c in self.shown(context) for h in core_hits(context, c)}

    def card(self, context: ReportContext, cluster: int) -> ClusterCard:
        members = np.flatnonzero(context.clusters == cluster)
        acts = context.activation[members]
        hits = core_hits(context, cluster)
        return ClusterCard(
            cluster=ref(cluster),
            size=len(members),
            sources=len({source_of(context.space.hit_ids[h]) for h in members}),
            composition=context.composition(members),
            low=float(acts.min()),
            high=float(acts.max()),
            median=float(np.median(acts)),
            confident=float((context.entropy[members] < 0.5).mean()),
            self_contained=float(
                context.contacts[cluster, cluster] / max(context.contacts[cluster].sum(), 1)
            ),
            neighbours=[(ref(k), s) for k, s in neighbours(context, cluster)],
            pattern=context.cluster_pattern(cluster),
            bucket_links=[
                (b.bucket.threshold, b.href)
                for b in context.buckets
                if list(b.bucket.clusters) == [cluster]
            ],
            strip=context.hit_view.strip(hits),
            patterns=context.hit_patterns(hits),
        )

    def view(self, context: ReportContext) -> dict[str, Any]:
        return {"cluster_cards": [self.card(context, c) for c in self.shown(context)]}


class GroupCards(Section):
    title = "Group clusters"
    guide = (
        GuideEntry(
            "Group clusters",
            "Hits that need the same set of clusters to cover 85% of their probability: the "
            "borders between clusters.",
        ),
    )
    template = "sections/group_cards.html.j2"

    def shown(self, context: ReportContext) -> list[LinkedBucket]:
        return [b for b in context.buckets if len(b.bucket.clusters) > 1]

    def sample(self, context: ReportContext, linked: LinkedBucket) -> list[int]:
        return [int(h) for h in ranked_sample(linked.bucket, context.probabilities, 10)]

    def hits(self, context: ReportContext) -> set[int]:
        return {h for b in self.shown(context) for h in self.sample(context, b)}

    def card(self, context: ReportContext, linked: LinkedBucket) -> GroupCard:
        b = linked.bucket
        hits = self.sample(context, linked)
        return GroupCard(
            name=b.name,
            href=linked.href,
            size=len(b.hits),
            clusters=[ref(c) for c in b.clusters],
            composition=context.composition(np.asarray(b.hits)),
            strip=context.hit_view.strip(hits),
            patterns=context.hit_patterns(hits),
        )

    def view(self, context: ReportContext) -> dict[str, Any]:
        return {"group_cards": [self.card(context, b) for b in self.shown(context)]}
