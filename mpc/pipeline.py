from dataclasses import dataclass
from typing import Any, Dict, Optional

import hdbscan
import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

from .bifiltration_processor import BifiltrationProcessor
from .clustering_utils import ClusteringUtils
from .filtration_builder import FiltrationBuilder


@dataclass
class BifiltrationResult:
    labels_combined: np.ndarray
    labels_filt1: np.ndarray
    labels_filt2: np.ndarray
    linkage_combined: np.ndarray
    linkage_filt1: np.ndarray
    linkage_filt2: np.ndarray
    processor: BifiltrationProcessor


def normalize_filtration(filt: list, target_max: float) -> list:
    max_val = max(v for _, v in filt)
    if max_val <= 0:
        raise ValueError("filtration has non-positive max; cannot normalize")
    ratio = target_max / max_val
    return [(s, v * ratio) for s, v in filt]


def _extract_labels(
    linkage: np.ndarray,
    method: str,
    n_points: int,
    n_clusters: Optional[int],
    min_size: int,
    persistence_threshold: Optional[float],
) -> np.ndarray:
    if method == "simplify":
        return ClusteringUtils.simplified_labels(
            linkage, n_clusters=n_clusters, min_size=min_size, n_points=n_points
        )
    if persistence_threshold is not None:
        return ClusteringUtils.persistence_cut(
            linkage,
            persistence_threshold=persistence_threshold,
            min_cluster_size=min_size,
        )
    return ClusteringUtils.persistence_cut(
        linkage, n_clusters=n_clusters, min_cluster_size=min_size,
    )


def run_bifiltration_edges(
    n_vertices: int,
    filt1: list,
    filt2: list,
    *,
    method: str = "simplify",
    min_size: Optional[int] = None,
    n_clusters: Optional[int] = None,
    persistence_threshold: Optional[float] = None,
    slope: float = 1.0,
) -> BifiltrationResult:
    if method not in {"simplify", "persistence"}:
        raise ValueError(f"unknown method {method!r}")
    if slope <= 0:
        raise ValueError("slope must be positive")

    N = int(n_vertices)
    if min_size is None:
        min_size = max(2, N // 20)

    filt2_norm = normalize_filtration(filt2, max(v for _, v in filt1))

    lm1 = ClusteringUtils.get_linkage_matrix(filt1, N)
    lm2 = ClusteringUtils.get_linkage_matrix(filt2_norm, N)
    k_for_individuals = n_clusters if n_clusters is not None else 1
    labels_f1 = fcluster(lm1, t=k_for_individuals, criterion="maxclust")
    labels_f2 = fcluster(lm2, t=k_for_individuals, criterion="maxclust")

    proc = BifiltrationProcessor(filt1, filt2_norm)
    slice_filt = proc.get_slice_optimized(
        f=(lambda x, a=slope: a * x),
        f_inverse=(lambda y, a=slope: y / a),
    )
    lm_c = ClusteringUtils.get_linkage_matrix(slice_filt, N)
    labels_c = _extract_labels(
        lm_c, method, N, n_clusters, min_size, persistence_threshold
    )

    return BifiltrationResult(
        labels_combined=labels_c,
        labels_filt1=labels_f1,
        labels_filt2=labels_f2,
        linkage_combined=lm_c,
        linkage_filt1=lm1,
        linkage_filt2=lm2,
        processor=proc,
    )


def run_bifiltration(
    data: np.ndarray,
    filt1: list,
    filt2: list,
    n_clusters: int,
    *,
    method: str = "simplify",
    min_size: Optional[int] = None,
    persistence_threshold: Optional[float] = None,
) -> BifiltrationResult:
    if method not in {"simplify", "persistence"}:
        raise ValueError(f"unknown method {method!r}")

    N = len(data)
    if min_size is None:
        min_size = max(2, N // 20)

    filt2_norm = normalize_filtration(filt2, max(v for _, v in filt1))

    lm1 = ClusteringUtils.get_linkage_matrix(filt1, N)
    lm2 = ClusteringUtils.get_linkage_matrix(filt2_norm, N)
    labels_f1 = fcluster(lm1, t=n_clusters, criterion="maxclust")
    labels_f2 = fcluster(lm2, t=n_clusters, criterion="maxclust")

    W1 = FiltrationBuilder.filtration_to_weight_matrix(filt1, N)
    W2 = FiltrationBuilder.filtration_to_weight_matrix(filt2_norm, N)
    W = BifiltrationProcessor.combined_weight_matrix(W1, W2, normalize=False)
    lm_c = ClusteringUtils.linkage_from_weight_matrix(W)
    labels_c = _extract_labels(
        lm_c, method, N, n_clusters, min_size, persistence_threshold
    )

    return BifiltrationResult(
        labels_combined=labels_c,
        labels_filt1=labels_f1,
        labels_filt2=labels_f2,
        linkage_combined=lm_c,
        linkage_filt1=lm1,
        linkage_filt2=lm2,
        processor=BifiltrationProcessor(filt1, filt2_norm),
    )


def run_standard_baselines(data: np.ndarray, n_clusters: int) -> Dict[str, np.ndarray]:
    return {
        "K-Means": KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10
        ).fit_predict(data),
        "Spectral": SpectralClustering(
            n_clusters=n_clusters, random_state=42, affinity="nearest_neighbors"
        ).fit_predict(data),
        "Ward": AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ).fit_predict(data),
        "Single Link": AgglomerativeClustering(
            n_clusters=n_clusters, linkage="single"
        ).fit_predict(data),
        "HDBSCAN": hdbscan.HDBSCAN(
            min_cluster_size=max(5, len(data) // 20)
        ).fit_predict(data),
    }


def evaluate_all(
    labels_true: np.ndarray, results_dict: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    return {
        name: ClusteringUtils.evaluate_clustering(labels_true, labels)
        for name, labels in results_dict.items()
    }


def graph_coherence(labels: np.ndarray, adjacency: np.ndarray) -> float:
    rows, cols = np.where(np.triu(adjacency, k=1) > 0)
    if len(rows) == 0:
        return 0.0
    return float(np.sum(labels[rows] == labels[cols]) / len(rows))


def cophenetic_correlation(linkage_matrix: np.ndarray, data: np.ndarray) -> float:
    c, _ = cophenet(linkage_matrix, pdist(data))
    return float(c)


def evaluate_unsupervised(
    labels: np.ndarray, data: np.ndarray, adjacency: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    labels = np.asarray(labels)
    nc = int(len(np.unique(labels)))
    out: Dict[str, Any] = {
        "nc": nc,
        "Sil": float(silhouette_score(data, labels)) if nc >= 2 else -1.0,
    }
    if adjacency is not None:
        out["GC"] = graph_coherence(labels, adjacency)
    return out
