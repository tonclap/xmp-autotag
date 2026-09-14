"""Tests for scripts/find_near_duplicates.py: union-find and threshold clustering.

The embedding index is faked with hand-built unit vectors, so the threshold and
transitivity behaviour is checked deterministically — no model, no archive, no
network.
"""
import math

import numpy as np
import pytest

import find_near_duplicates as fnd


# --- UnionFind -----------------------------------------------------------

def test_unionfind_transitive_connectivity():
    uf = fnd.UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)  # one component, no direct 0-2 edge
    assert uf.find(0) != uf.find(3)


def test_unionfind_union_is_idempotent():
    uf = fnd.UnionFind(3)
    uf.union(0, 1)
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)


# --- find_clusters -------------------------------------------------------

def _unit_vec(theta):
    return [math.cos(theta), math.sin(theta)]


@pytest.fixture
def synthetic_index(monkeypatch):
    """Two folders:

    A: p1 and p2 are 0.95 similar (a cluster of 2); p3 is nearly orthogonal.
    B: p4~p5 and p5~p6 are each 0.95, while p4~p6 is only 0.805 — they must end
       up in ONE cluster of 3 through transitivity.
    p1 and p4 have identical vectors but live in different folders, so they must
    never be merged.
    """
    delta = math.acos(0.95)
    vectors = np.array([
        _unit_vec(0.0),          # p1 (A)
        _unit_vec(delta),        # p2 (A)
        _unit_vec(math.pi / 2),  # p3 (A)
        _unit_vec(0.0),          # p4 (B), identical to p1
        _unit_vec(delta),        # p5 (B)
        _unit_vec(2 * delta),    # p6 (B)
    ], dtype=np.float64)
    meta = [
        {"path": "A/p1.jpg", "description": "p1"},
        {"path": "A/p2.jpg", "description": "p2"},
        {"path": "A/p3.jpg", "description": "p3"},
        {"path": "B/p4.jpg", "description": "p4"},
        {"path": "B/p5.jpg", "description": "p5"},
        {"path": "B/p6.jpg", "description": "p6"},
    ]
    monkeypatch.setattr(fnd.sc, "get_index", lambda: (vectors, meta, 0))
    return vectors, meta


def test_clusters_stay_inside_a_folder_and_follow_the_chain(synthetic_index):
    clusters, compared_pairs, total = fnd.find_clusters(threshold=0.93, min_cluster=2)

    assert total == 6
    # C(3,2) within A plus C(3,2) within B; no cross-folder pairs at all.
    assert compared_pairs == 6
    assert sorted(c["size"] for c in clusters) == [2, 3]

    by_size = {c["size"]: c for c in clusters}
    paths_a = {p["path"] for p in by_size[2]["photos"]}
    paths_b = {p["path"] for p in by_size[3]["photos"]}

    assert paths_a == {"A/p1.jpg", "A/p2.jpg"}
    assert paths_b == {"B/p4.jpg", "B/p5.jpg", "B/p6.jpg"}
    # Averages count only the edges above the threshold, so the 0.805 pair is
    # excluded from the chain's average.
    assert by_size[2]["avg_similarity"] == pytest.approx(0.95, abs=1e-3)
    assert by_size[3]["avg_similarity"] == pytest.approx(0.95, abs=1e-3)
    # Identical vectors in different folders stay apart; an unpaired image is
    # not reported at all.
    assert "B/p4.jpg" not in paths_a and "A/p1.jpg" not in paths_b
    assert "A/p3.jpg" not in paths_a | paths_b


def test_min_cluster_filters_small_groups(synthetic_index):
    clusters, _, _ = fnd.find_clusters(threshold=0.93, min_cluster=3)
    assert len(clusters) == 1
    assert clusters[0]["size"] == 3


def test_a_higher_threshold_breaks_the_chain(synthetic_index):
    clusters, _, _ = fnd.find_clusters(threshold=0.97, min_cluster=2)
    assert clusters == []
