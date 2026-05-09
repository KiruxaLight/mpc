from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage as _scipy_linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


@dataclass(frozen=True)
class _LinkageMeta:
    n: int
    left: np.ndarray
    right: np.ndarray
    dist: np.ndarray
    birth: np.ndarray
    subtree_size: np.ndarray
    any_leaf: np.ndarray


def _linkage_metadata(Z: np.ndarray) -> _LinkageMeta:
    Z = np.asarray(Z, dtype=float)
    m = Z.shape[0]
    n = m + 1
    left = Z[:, 0].astype(int)
    right = Z[:, 1].astype(int)
    dist = Z[:, 2].astype(float)

    birth = np.zeros(n + m, dtype=float)
    birth[n:] = dist

    subtree_size = np.ones(n + m, dtype=int)
    subtree_size[n:] = Z[:, 3].astype(int)

    any_leaf = np.arange(n + m, dtype=int)
    for i in range(m):
        any_leaf[n + i] = any_leaf[left[i]]

    return _LinkageMeta(n, left, right, dist, birth, subtree_size, any_leaf)


class ClusteringUtils:
    @staticmethod
    def format_filtration(filtration: list) -> List[Tuple[int, int, float]]:
        return [(int(s[0]), int(s[1]), float(v))
                for s, v in filtration if len(s) == 2]

    @staticmethod
    def get_linkage_matrix(filtration: list, n_points: int) -> np.ndarray:
        edges = ClusteringUtils.format_filtration(filtration)

        uf = UnionFind(n_points)
        root_cluster_id = list(range(n_points))
        root_cluster_size = [1] * n_points
        linkage: list = []
        next_id = n_points

        for u, v, d in edges:
            ru, rv = uf.find(u), uf.find(v)
            if ru == rv:
                continue
            merged_size = root_cluster_size[ru] + root_cluster_size[rv]
            linkage.append([root_cluster_id[ru], root_cluster_id[rv], d, merged_size])
            uf.union(ru, rv)
            new_root = uf.find(ru)
            root_cluster_id[new_root] = next_id
            root_cluster_size[new_root] = merged_size
            next_id += 1

        return np.asarray(linkage, dtype=float)

    @staticmethod
    def linkage_from_weight_matrix(W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        np.fill_diagonal(W, 0.0)
        return _scipy_linkage(squareform(W, checks=False), method="single")

    @staticmethod
    def get_labels(
        linkage_matrix: np.ndarray,
        n_clusters: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        if (n_clusters is None) == (threshold is None):
            raise ValueError("specify exactly one of n_clusters, threshold")
        if n_clusters is not None:
            return fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        return fcluster(linkage_matrix, t=threshold, criterion="distance")

    @staticmethod
    def evaluate_clustering(labels_true: Sequence[int],
                            labels_pred: Sequence[int]) -> dict:
        return {
            "ARI": float(adjusted_rand_score(labels_true, labels_pred)),
            "NMI": float(normalized_mutual_info_score(labels_true, labels_pred)),
            "V-measure": float(v_measure_score(labels_true, labels_pred)),
        }

    @staticmethod
    def merge_small_clusters_grid(labels_2d: np.ndarray, min_size: int) -> np.ndarray:
        labels = labels_2d.copy()
        rows, cols = labels.shape
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))

        while True:
            unique, counts = np.unique(labels, return_counts=True)
            sizes = dict(zip(unique.tolist(), counts.tolist()))
            small = [c for c, s in sizes.items() if s < min_size and s > 0]
            if not small:
                return labels

            changed = False
            for sc in small:
                if sizes.get(sc, 0) == 0:
                    continue
                mask = labels == sc
                ys, xs = np.where(mask)
                large, any_nb = set(), set()
                for y, x in zip(ys, xs):
                    for dy, dx in deltas:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            nc = int(labels[ny, nx])
                            if nc != sc:
                                any_nb.add(nc)
                                if sizes.get(nc, 0) >= min_size:
                                    large.add(nc)
                target = large or any_nb
                if target:
                    best = max(target, key=lambda c: sizes.get(c, 0))
                    labels[mask] = best
                    sizes[best] += sizes[sc]
                    sizes[sc] = 0
                    changed = True
            if not changed:
                return labels

    @staticmethod
    def simplify_linkage(
        linkage_matrix: np.ndarray, min_size: int
    ) -> Tuple[np.ndarray, List[List[int]]]:
        Z = np.asarray(linkage_matrix, dtype=float)
        if Z.size == 0:
            return np.zeros((0, 4)), [[0]]

        n = int(Z.shape[0]) + 1
        subtree_sz = np.ones(n + len(Z), dtype=int)
        for i in range(len(Z)):
            left, right = int(Z[i, 0]), int(Z[i, 1])
            subtree_sz[n + i] = subtree_sz[left] + subtree_sz[right]

        parent = list(range(n + len(Z)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        group_members: dict = {i: [i] for i in range(n)}
        group_sz = [0] * (n + len(Z))
        for i in range(n):
            group_sz[i] = 1

        kept: list = []
        kept_node_set: set = set()
        kept_children: dict = {}

        def find_macro_leaf(x: int) -> int:
            while x in kept_node_set:
                clr, crr = kept_children[x]
                x = clr if group_sz[clr] >= group_sz[crr] else crr
            return x

        for i in range(len(Z)):
            left, right = int(Z[i, 0]), int(Z[i, 1])
            dist = float(Z[i, 2])
            node_id = n + i
            lr, rr = find(left), find(right)
            left_sz = int(subtree_sz[left])
            right_sz = int(subtree_sz[right])

            if left_sz >= min_size and right_sz >= min_size:
                parent[node_id] = node_id
                kept.append((node_id, lr, rr, dist))
                kept_node_set.add(node_id)
                kept_children[node_id] = (lr, rr)
            else:
                if left_sz >= right_sz:
                    big_repr, small_repr = lr, rr
                else:
                    big_repr, small_repr = rr, lr
                big_leaf = find_macro_leaf(big_repr)
                small_leaf = find_macro_leaf(small_repr)
                if big_leaf != small_leaf:
                    group_members[big_leaf] = (
                        group_members[big_leaf] + group_members[small_leaf]
                    )
                    group_sz[big_leaf] += group_sz[small_leaf]
                    group_sz[small_leaf] = 0
                    del group_members[small_leaf]
                parent[small_repr] = big_repr
                parent[node_id] = big_repr

        if not kept:
            root = find_macro_leaf(find(0))
            return np.zeros((0, 4)), [group_members[root]]

        leaf_reprs = sorted(group_members.keys())
        leaf_id_map = {nid: j for j, nid in enumerate(leaf_reprs)}
        leaf_members = [group_members[nid] for nid in leaf_reprs]

        node_new_id = dict(leaf_id_map)
        leaf_count = {nid: 1 for nid in leaf_reprs}
        next_id = len(leaf_reprs)
        new_rows: list = []
        for node_id, lr, rr, dist in kept:
            nl = node_new_id[lr]
            nr = node_new_id[rr]
            leaf_count[node_id] = leaf_count[lr] + leaf_count[rr]
            new_rows.append([nl, nr, dist, leaf_count[node_id]])
            node_new_id[node_id] = next_id
            next_id += 1

        return np.asarray(new_rows, dtype=float), leaf_members

    @staticmethod
    def expand_simplified_labels(
        macro_labels: np.ndarray,
        leaf_members: List[List[int]],
        n_points: int,
    ) -> np.ndarray:
        labels = np.zeros(n_points, dtype=int)
        for j, pts in enumerate(leaf_members):
            labels[pts] = macro_labels[j]
        return labels

    @staticmethod
    def simplified_labels(
        linkage_matrix: np.ndarray,
        n_clusters: Optional[int] = None,
        min_size: Optional[int] = None,
        n_points: Optional[int] = None,
    ) -> np.ndarray:
        Z = np.asarray(linkage_matrix, dtype=float)
        N = n_points if n_points is not None else int(Z.shape[0] + 1)
        if min_size is None:
            min_size = max(2, N // 20)

        sim_Z, leaf_members = ClusteringUtils.simplify_linkage(Z, min_size)
        if n_clusters is None:
            macro = np.arange(1, len(leaf_members) + 1, dtype=int)
            return ClusteringUtils.expand_simplified_labels(macro, leaf_members, N)
        if len(sim_Z) == 0:
            return np.ones(N, dtype=int)
        k = min(n_clusters, len(leaf_members))
        macro = fcluster(sim_Z, t=k, criterion="maxclust")
        return ClusteringUtils.expand_simplified_labels(macro, leaf_members, N)

    @staticmethod
    def compute_merge_persistences(linkage_matrix: np.ndarray) -> np.ndarray:
        Z = np.asarray(linkage_matrix, dtype=float)
        if Z.size == 0:
            return np.zeros(0, dtype=float)
        meta = _linkage_metadata(Z)
        return meta.dist - np.maximum(meta.birth[meta.left], meta.birth[meta.right])

    @staticmethod
    def persistence_cut(
        linkage_matrix: np.ndarray,
        n_clusters: Optional[int] = None,
        persistence_threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
    ) -> np.ndarray:
        if (n_clusters is None) == (persistence_threshold is None):
            raise ValueError(
                "specify exactly one of n_clusters, persistence_threshold"
            )
        Z = np.asarray(linkage_matrix, dtype=float)
        if Z.size == 0:
            return np.zeros(1, dtype=int)

        meta = _linkage_metadata(Z)
        if min_cluster_size is None:
            min_cluster_size = max(2, meta.n // 20)
        m = len(meta.left)
        persistences = meta.dist - np.maximum(
            meta.birth[meta.left], meta.birth[meta.right]
        )
        eligible = (
            (meta.subtree_size[meta.left] >= min_cluster_size)
            & (meta.subtree_size[meta.right] >= min_cluster_size)
        )

        if n_clusters is not None:
            if not 1 <= n_clusters <= meta.n:
                raise ValueError(
                    f"n_clusters must be in [1, {meta.n}], got {n_clusters}"
                )
            skip_mask = np.zeros(m, dtype=bool)
            n_skip = n_clusters - 1
            if n_skip > 0:
                masked = np.where(eligible, persistences, -np.inf)
                top = np.argsort(masked)[::-1][:n_skip]
                skip_mask[top[np.isfinite(masked[top])]] = True
        else:
            skip_mask = eligible & (persistences >= persistence_threshold)

        uf = UnionFind(meta.n)
        to_merge = np.where(~skip_mask)[0]
        for i in to_merge:
            uf.union(int(meta.any_leaf[meta.left[i]]),
                     int(meta.any_leaf[meta.right[i]]))

        roots = np.fromiter((uf.find(i) for i in range(meta.n)),
                            dtype=int, count=meta.n)
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(int)

    @staticmethod
    def segment_superpixels_otsu(
        linkage_matrix: np.ndarray,
        image: np.ndarray,
        n_superpixels: int,
        min_pix: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from skimage.filters import threshold_otsu

        rows, cols = image.shape
        labels = fcluster(linkage_matrix, t=n_superpixels, criterion="maxclust")
        labels_2d = labels.reshape(rows, cols)
        labels_2d = ClusteringUtils.merge_small_clusters_grid(labels_2d, min_pix)
        superpixels = labels_2d.copy()

        unique, sizes = np.unique(labels_2d, return_counts=True)
        flat_image = image.ravel()
        flat_labels = labels_2d.ravel()
        mean_brightness = np.array([
            flat_image[flat_labels == lab].mean() for lab in unique
        ])
        thresh = threshold_otsu(np.repeat(mean_brightness, sizes))

        brightness_by_label = dict(zip(unique.tolist(), mean_brightness.tolist()))
        result = np.zeros_like(labels_2d)
        for lab in unique:
            if brightness_by_label[int(lab)] > thresh:
                result[labels_2d == lab] = 1
        return result, superpixels
