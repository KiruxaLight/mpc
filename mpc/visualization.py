from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_clusters_2d(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    cmap: str = "tab20",
    s: int = 30,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, s=s,
               edgecolors="k", linewidths=0.2)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_clusters_3d(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    cmap: str = "tab20",
    s: int = 25,
) -> plt.Axes:
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=labels, cmap=cmap, s=s, edgecolors="k", linewidths=0.2)
    ax.set_title(title)
    return ax


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Dendrogram",
    ax: Optional[plt.Axes] = None,
    truncate_p: int = 30,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    n_samples = linkage_matrix.shape[0] + 1
    if n_samples > 500:
        dendrogram(linkage_matrix, ax=ax, no_labels=True,
                   truncate_mode="lastp", p=truncate_p)
    else:
        dendrogram(linkage_matrix, ax=ax, no_labels=True)
    if threshold is not None:
        ax.axhline(threshold, color="r", linestyle="--",
                   label=f"threshold = {threshold:.2f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Data points")
    ax.set_ylabel("Distance")
    return ax


def plot_comparison_grid(
    data: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    title: str = "",
    ncols: int = 3,
    figsize_per_plot: tuple = (4.5, 4.0),
) -> plt.Figure:
    items = list(labels_dict.items())
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, labels) in zip(axes, items):
        plot_clusters_2d(data, labels, title=name, ax=ax)
    for ax in axes[len(items):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    return fig


def plot_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    method_names = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    values = np.array([[metrics[m][k] for k in metric_names] for m in method_names])

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.4 * len(method_names) + 4), 3))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=9)
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return ax
