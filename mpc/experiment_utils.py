from collections import Counter
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def align_hungarian(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    u_pred = sorted(np.unique(pred).tolist())
    u_gt = sorted(np.unique(gt).tolist())
    p1 = np.zeros_like(pred)
    for i, up in enumerate(u_pred):
        p1[pred == up] = i + 1
    g2i = {g: i + 1 for i, g in enumerate(u_gt)}
    gtn = np.array([g2i[int(x)] for x in gt])
    size = max(len(u_pred), len(u_gt))
    cost = np.zeros((size, size), dtype=np.int64)
    for i in range(len(u_pred)):
        for j in range(len(u_gt)):
            cost[i, j] = -int(((p1 == i + 1) & (gtn == j + 1)).sum())
    r, c = linear_sum_assignment(cost)
    remap = {int(r[t] + 1): int(c[t] + 1)
             for t in range(len(r)) if r[t] < len(u_pred)}
    out = np.zeros_like(pred)
    for s, d in remap.items():
        out[p1 == s] = d
    return out, gtn


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    la, gtn = align_hungarian(pred, gt)
    total = sum(Counter(gtn[la == k].tolist()).most_common(1)[0][1]
                for k in np.unique(la))
    metrics = dict(
        ari=float(adjusted_rand_score(gtn, la)),
        nmi=float(normalized_mutual_info_score(gtn, la)),
        vm=float(v_measure_score(gtn, la)),
        f1m=float(f1_score(gtn, la, average="macro", zero_division=0)),
        purity=float(total / len(la)),
    )
    return metrics, la
