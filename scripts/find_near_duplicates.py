"""Find near-duplicate frames: burst shots, not byte-identical copies.

    python scripts/find_near_duplicates.py
    python scripts/find_near_duplicates.py --threshold 0.9 --min-cluster 3

Reuses the description embeddings that power search — there is no separate visual
model here. That makes this a proxy rather than true visual similarity: frames of
one burst usually get nearly identical descriptions (same scene, same setting,
seconds apart), but the method can also merge visually different images that
happen to be described alike, and miss similar frames described with a different
emphasis. It only reports clusters; nothing is ever deleted.

Comparison happens strictly within one folder. A burst is physically stored
together, so this both cuts the work from N^2 over the whole archive to the sum
of squares per folder, and avoids merging unrelated folders with similar subjects.
Clusters are connected components, not pairs: a burst of five frames must come
out as one cluster even if its first and last frame no longer resemble each other.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import config
import search_core as sc

OUT_PATH = config.DUPLICATES_PATH
# Calibrated by hand against real bursts across the 0.90-1.0 range, with
# consecutive frame numbers and capture seconds as independent confirmation.
# Lower merges different scenes; higher misses bursts whose descriptions vary.
DEFAULT_THRESHOLD = 0.93


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_clusters(threshold, min_cluster):
    vectors, meta, _ = sc.get_index()
    if vectors is None or not meta:
        sys.exit("no index yet - run: python scripts/build_search_index.py")

    by_folder = defaultdict(list)
    for i, m in enumerate(meta):
        by_folder[str(Path(m["path"]).parent)].append(i)

    uf = UnionFind(len(meta))
    pair_scores = {}
    compared_pairs = 0
    for indices in by_folder.values():
        if len(indices) < 2:
            continue
        sub = vectors[indices]
        similarity = sub @ sub.T
        n = len(indices)
        for a in range(n):
            for b in range(a + 1, n):
                compared_pairs += 1
                score = float(similarity[a, b])
                if score >= threshold:
                    gi, gj = indices[a], indices[b]
                    uf.union(gi, gj)
                    pair_scores[(gi, gj)] = score

    groups = defaultdict(list)
    for i in range(len(meta)):
        groups[uf.find(i)].append(i)

    # Edges bucketed by the cluster they ended up in. Scanning the whole pair
    # table once per cluster instead is O(clusters x pairs) — invisible on a
    # handful of bursts, and the dominant cost on an archive where most folders
    # produce one.
    edges_by_root = defaultdict(list)
    for (a, _b), score in pair_scores.items():
        edges_by_root[uf.find(a)].append(score)

    clusters = []
    for root, members in groups.items():
        if len(members) < min_cluster:
            continue
        # Average over the edges that actually passed the threshold: a cluster
        # can be a chain A~B~C with no direct A~C edge.
        edges = edges_by_root[root]
        clusters.append({
            "size": len(members),
            "avg_similarity": round(sum(edges) / len(edges), 4) if edges else None,
            "photos": sorted(
                [{"path": meta[i]["path"], "description": meta[i]["description"]} for i in members],
                key=lambda item: item["path"],
            ),
        })
    clusters.sort(key=lambda c: (-c["size"], -(c["avg_similarity"] or 0)))
    return clusters, compared_pairs, len(meta)


def main():
    ap = argparse.ArgumentParser(description="Cluster near-duplicate frames within each folder")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-cluster", type=int, default=2)
    args = ap.parse_args()

    clusters, compared_pairs, total = find_clusters(args.threshold, args.min_cluster)
    clustered = sum(c["size"] for c in clusters)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "threshold": args.threshold,
        "total_photos": total,
        "compared_pairs": compared_pairs,
        "clusters": clusters,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"threshold {args.threshold}, pairs compared within folders: {compared_pairs}")
    print(f"clusters: {len(clusters)}, images in clusters: {clustered}/{total}")
    print(f"report: {OUT_PATH}")


if __name__ == "__main__":
    main()
