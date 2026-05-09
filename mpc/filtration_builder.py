from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures


def _tie_break_eps(max_val: float) -> float:
    return max(max_val * np.finfo(float).eps * 4.0, np.finfo(float).tiny)


def assemble_filtration(
    n_vertices: int,
    edges: Iterable[Tuple[int, int]],
    weights: Sequence[float],
) -> list:
    weights = np.asarray(weights, dtype=float)
    edges = np.asarray(list(edges), dtype=int)
    if edges.shape[0] != len(weights):
        raise ValueError("edges and weights must be the same length")

    order = np.lexsort((edges[:, 1], edges[:, 0], weights))
    eps = _tie_break_eps(float(np.abs(weights).max()) if weights.size else 1.0)

    filtration: list = [([int(i)], 0.0) for i in range(n_vertices)]
    prev = -np.inf
    for idx in order:
        w = float(weights[idx])
        if w <= prev:
            w = prev + eps
        i, j = int(edges[idx, 0]), int(edges[idx, 1])
        filtration.append(([i, j], w))
        prev = w
    return filtration


def _dense_to_rips(weights: np.ndarray) -> list:
    weights = np.asarray(weights, dtype=float)
    n = weights.shape[0]
    if weights.shape != (n, n):
        raise ValueError(f"weights must be square; got {weights.shape}")
    iu, ju = np.triu_indices(n, k=1)
    return assemble_filtration(n, zip(iu.tolist(), ju.tolist()), weights[iu, ju])


def _complementary_density_weights(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) >= 2:
        two_largest = np.partition(v, -2)[-2:]
        max_sum = float(two_largest.sum())
    else:
        max_sum = float(v.max()) * 2.0 if len(v) else 0.0
    weights = max_sum - (v[:, None] + v[None, :])
    np.fill_diagonal(weights, 0.0)
    return weights


@dataclass
class FiltrationBuilder:
    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)

    def get_filtration_from_scipy_dist(self, metric: str = "euclidean") -> list:
        return _dense_to_rips(cdist(self.data, self.data, metric))

    def get_filtration_from_density(
        self, kernel: str = "gaussian", bandwidth: float = 0.3
    ) -> list:
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(self.data)
        density = np.exp(kde.score_samples(self.data))
        diff = np.abs(density[:, None] - density[None, :])
        return _dense_to_rips(diff)

    def get_filtration_from_knn(self, k: int = 5) -> list:
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(self.data))).fit(self.data)
        distances, _ = nbrs.kneighbors(self.data)
        return _dense_to_rips(_complementary_density_weights(distances[:, -1]))

    def get_filtration_from_local_density(self, r: float) -> list:
        tree = KDTree(self.data)
        counts = tree.query_radius(self.data, r, count_only=True).astype(float)
        return _dense_to_rips(_complementary_density_weights(counts))

    def get_filtration_from_christoffel(self, degree: int = 2) -> list:
        n = len(self.data)
        V = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(self.data)
        s = V.shape[1]
        M_inv = np.linalg.pinv((V.T @ V) / n)
        Q = np.einsum("ij,jk,ik->i", V, M_inv, V)
        density = s / np.maximum(Q, 1e-15)
        return _dense_to_rips(_complementary_density_weights(density))

    def get_filtration_from_intrinsic_dim(
        self,
        k_neighbors: int = 10,
        var_threshold: float = 0.95,
        connectivity_penalty: float = 8.0,
        dim_weight: float = 10.0,
        rng_seed: int = 0,
    ) -> list:
        n = len(self.data)
        k_neighbors = min(k_neighbors, n - 1)

        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(self.data)
        _, indices = nbrs.kneighbors(self.data)

        dims = np.empty(n, dtype=float)
        for i in range(n):
            pca = PCA().fit(self.data[indices[i, 1:]])
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            dims[i] = float(np.searchsorted(cum_var, var_threshold) + 1)

        wide_k = min(2 * k_neighbors, n - 1)
        _, idx_wide = NearestNeighbors(n_neighbors=wide_k + 1).fit(self.data).kneighbors(self.data)
        neighbor_mask = np.zeros((n, n), dtype=bool)
        rows = np.repeat(np.arange(n), wide_k)
        neighbor_mask[rows, idx_wide[:, 1:].ravel()] = True
        neighbor_mask |= neighbor_mask.T

        rng = np.random.RandomState(rng_seed)
        noise = rng.rand(n, n) * 1e-6
        noise = np.triu(noise, k=1)
        noise = noise + noise.T

        dim_diff = np.abs(dims[:, None] - dims[None, :])
        same_dim = dim_diff < 1e-9
        weights = dim_weight * dim_diff + noise
        weights += connectivity_penalty * (same_dim & ~neighbor_mask)
        np.fill_diagonal(weights, 0.0)
        return _dense_to_rips(weights)

    @staticmethod
    def estimate_local_dim_mle(
        data: np.ndarray,
        k: int = 15,
        dim_max: Optional[int] = None,
        flat_percentile: float = 0.0,
        jitter_seed: int = 0,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        n, D = data.shape
        dim_max = dim_max if dim_max is not None else D

        rng = np.random.RandomState(jitter_seed)
        norms = np.linalg.norm(data, axis=1)
        median_norm = float(np.median(norms[norms > 0])) if np.any(norms > 0) else 1.0
        jitter_scale = max(median_norm * 1e-8, 1e-15)
        data_j = data + rng.normal(0.0, jitter_scale, data.shape)

        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(data_j)
        distances, _ = nbrs.kneighbors(data_j)

        T_k = np.maximum(distances[:, -1], 1e-15)[:, None]
        T_j = np.maximum(distances[:, 1:-1], 1e-15)
        log_ratios = np.log(T_k / T_j)
        valid = T_j > 1e-14
        log_sum = np.sum(log_ratios * valid, axis=1)
        count = np.sum(valid, axis=1)

        dims = np.where(
            (count > 1) & (log_sum > 1e-15),
            (count - 1) / np.maximum(log_sum, 1e-15),
            1.0,
        )
        if flat_percentile > 0:
            norm_thresh = float(np.percentile(norms, flat_percentile))
            dims[norms <= norm_thresh] = 1.0
        return np.clip(dims, 1.0, dim_max)

    def get_filtration_from_intdim_mle(
        self, k: int = 15, flat_percentile: float = 0.0
    ) -> list:
        dims = FiltrationBuilder.estimate_local_dim_mle(
            self.data, k=k, flat_percentile=flat_percentile
        )
        return _dense_to_rips(np.abs(dims[:, None] - dims[None, :]))

    def get_filtration_from_tangent_direction(self, k: int = 5) -> list:
        n, d = self.data.shape
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(self.data)
        _, indices = nbrs.kneighbors(self.data)

        tangents = np.empty((n, d), dtype=float)
        for i in range(n):
            pca = PCA(n_components=min(d, k)).fit(self.data[indices[i, 1:]])
            tangents[i] = pca.components_[0]

        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.where(norms == 0, 1.0, norms)
        weights = 1.0 - np.clip(np.abs(tangents @ tangents.T), 0.0, 1.0)
        np.fill_diagonal(weights, 0.0)
        return _dense_to_rips(weights)

    def get_filtration_from_curvature(self, k: int = 8) -> list:
        if self.data.shape[1] != 2:
            raise ValueError("curvature filtration expects 2-D data")
        n = len(self.data)
        k = min(k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(self.data)
        _, idx = nbrs.kneighbors(self.data)

        curv = np.zeros(n, dtype=float)
        for i in range(n):
            lam = PCA(n_components=2).fit(self.data[idx[i, 1:]]).explained_variance_
            s = lam.sum()
            if s > 0:
                curv[i] = 1.0 - (lam[0] / s)
        weights = curv[:, None] + curv[None, :]
        np.fill_diagonal(weights, 0.0)
        return _dense_to_rips(weights)

    @staticmethod
    def get_scalar_edge_filtration(
        scalar: np.ndarray, edges: np.ndarray
    ) -> list:
        scalar = np.asarray(scalar, dtype=float)
        edges = np.asarray(edges, dtype=int)
        weights = np.abs(scalar[edges[:, 0]] - scalar[edges[:, 1]])
        return assemble_filtration(len(scalar), edges, weights)

    @staticmethod
    def _grid_edges(rows: int, cols: int, connectivity: int = 4) -> np.ndarray:
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
        r_idx, c_idx = np.indices((rows, cols))
        flat_idx = r_idx * cols + c_idx

        edge_lists: list = [
            np.column_stack([flat_idx[:, :-1].ravel(), flat_idx[:, 1:].ravel()]),
            np.column_stack([flat_idx[:-1, :].ravel(), flat_idx[1:, :].ravel()]),
        ]
        if connectivity == 8:
            edge_lists.append(np.column_stack([flat_idx[:-1, :-1].ravel(),
                                               flat_idx[1:, 1:].ravel()]))
            edge_lists.append(np.column_stack([flat_idx[:-1, 1:].ravel(),
                                               flat_idx[1:, :-1].ravel()]))
        return np.concatenate(edge_lists, axis=0)

    @classmethod
    def _image_to_gray(cls, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=float)
        if image.ndim == 3:
            image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        return image

    @classmethod
    def build_grid_brightness_filtration(
        cls, image: np.ndarray, connectivity: int = 4
    ) -> Tuple[list, int, int, int]:
        image = cls._image_to_gray(image)
        rows, cols = image.shape
        n = rows * cols
        b = image.ravel()
        edges = cls._grid_edges(rows, cols, connectivity)
        weights = np.abs(b[edges[:, 0]] - b[edges[:, 1]])
        return assemble_filtration(n, edges, weights), n, rows, cols

    @classmethod
    def build_grid_gradient_filtration(
        cls, image: np.ndarray, connectivity: int = 4
    ) -> Tuple[list, int, int, int]:
        image = cls._image_to_gray(image)
        rows, cols = image.shape
        n = rows * cols
        gy, gx = np.gradient(image)
        grad = np.sqrt(gx ** 2 + gy ** 2).ravel()
        edges = cls._grid_edges(rows, cols, connectivity)
        weights = np.maximum(grad[edges[:, 0]], grad[edges[:, 1]])
        return assemble_filtration(n, edges, weights), n, rows, cols

    @staticmethod
    def filtration_to_weight_matrix(filtration: list, n: int) -> np.ndarray:
        W = np.zeros((n, n), dtype=float)
        for simplex, value in filtration:
            if len(simplex) == 1:
                if value > 0:
                    raise ValueError(
                        "filtration has a vertex with value > 0 — not Rips-type"
                    )
            elif len(simplex) == 2:
                i, j = int(simplex[0]), int(simplex[1])
                W[i, j] = value
                W[j, i] = value
        return W
