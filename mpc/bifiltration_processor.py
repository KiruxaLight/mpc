from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class BifiltrationProcessor:
    def __init__(self, filtration1: list, filtration2: list) -> None:
        self.filtration1 = filtration1
        self.filtration2 = filtration2

    def _indexed_filtrations(self) -> Tuple[dict, dict]:
        f1 = {tuple(s): (float(v), i) for i, (s, v) in enumerate(self.filtration1)}
        f2 = {tuple(s): (float(v), i) for i, (s, v) in enumerate(self.filtration2)}
        return f1, f2

    def critical_points(self) -> Tuple[np.ndarray, dict, dict, dict, dict]:
        f1, f2 = self._indexed_filtrations()
        n1, n2 = len(self.filtration1), len(self.filtration2)
        grid = np.full((n1, n2), -1, dtype=np.int8)
        f1_rev = {i: list(s) for s, (_, i) in f1.items()}
        f2_rev = {i: list(s) for s, (_, i) in f2.items()}

        for s, (_, i1) in f1.items():
            if s in f2:
                grid[i1, f2[s][1]] = 0

        for i in range(1, n1):
            for j in range(1, n2):
                if grid[i, j] == -1 and grid[i - 1, j] == 0 and grid[i, j - 1] == 0:
                    grid[i, j] = 1
        return grid, f1_rev, f2_rev, f1, f2

    def get_raw_critical_points(self) -> List[Tuple[float, float, list]]:
        f1, f2 = self._indexed_filtrations()
        return [(v1, f2[s][0], list(s))
                for s, (v1, _) in f1.items() if s in f2]

    def get_slice_optimized(
        self,
        f: Callable[[float], float] = lambda x: x,
        f_inverse: Callable[[float], float] = lambda y: y,
    ) -> list:
        f1, f2 = self._indexed_filtrations()
        prepared: list = []
        for s, (v1, _) in f1.items():
            if s not in f2:
                continue
            v2 = f2[s][0]
            x_line = f_inverse(v2)
            y_line = f(v1)
            if y_line <= v2:
                proj = v2 + x_line
                dist = abs(x_line - v1)
            else:
                proj = y_line + v1
                dist = abs(y_line - v2)
            prepared.append((list(s), proj, dist))

        prepared.sort(key=lambda row: (row[1], row[2]))
        return [(row[0], row[1]) for row in prepared]

    def get_slice(
        self,
        f: Callable[[float], float] = lambda x: x,
        f_inverse: Callable[[float], float] = lambda y: y,
    ) -> list:
        grid, f1_rev, f2_rev, _, _ = self.critical_points()
        n1, n2 = grid.shape
        prepared: list = []
        for x_idx in range(n1):
            for y_idx in range(n2):
                cell = grid[x_idx, y_idx]
                if cell == -1:
                    continue
                v1 = self.filtration1[x_idx][1]
                v2 = self.filtration2[y_idx][1]
                y_line = f(v1)
                x_line = f_inverse(v2)
                if y_line <= v2:
                    src = f1_rev.get(x_idx if cell == 0 else x_idx - 1)
                    proj = v2 + x_line
                    dist = x_line - v1
                else:
                    src = f2_rev.get(y_idx if cell == 0 else y_idx - 1)
                    proj = y_line + v1
                    dist = y_line - v2
                if src is not None:
                    prepared.append((list(src), proj, abs(dist)))

        prepared.sort(key=lambda row: (row[1], row[2]))
        out, last_proj = [], None
        for s, p, _ in prepared:
            if last_proj is None or p != last_proj:
                out.append((s, p))
                last_proj = p
        return out

    @staticmethod
    def combined_weight_matrix(
        W1: np.ndarray, W2: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        W1 = np.asarray(W1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        if W1.shape != W2.shape:
            raise ValueError(f"shape mismatch: {W1.shape} vs {W2.shape}")
        if normalize:
            m1, m2 = float(W1.max()), float(W2.max())
            if m2 > 0:
                W2 = W2 * (m1 / m2)
        combined = np.maximum(W1, W2)
        np.fill_diagonal(combined, 0.0)
        return combined

    def plot_critical_points_2d(
        self,
        title: str = "Critical Points",
        color: str = "tab:blue",
        marker_size: int = 40,
        plot_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        func_x_range: Optional[tuple] = None,
        show_labels: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        points = self.get_raw_critical_points()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=marker_size, c=color, edgecolors="black",
                   linewidths=0.4, label="Critical points")
        if plot_function is not None and xs:
            x_lo, x_hi = func_x_range if func_x_range else (min(xs), max(xs))
            xs_line = np.linspace(x_lo, x_hi, 200)
            ax.plot(xs_line, plot_function(xs_line), "r--", label="Slicing line")
        if show_labels:
            for x, y, s in points:
                ax.text(x, y, str(s), fontsize=7)
        ax.set_xlabel("Filtration 1 value")
        ax.set_ylabel("Filtration 2 value")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        return ax
