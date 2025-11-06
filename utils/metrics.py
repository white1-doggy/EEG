from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")


def confidence_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    mean = values.mean()
    se = stats.sem(values)
    if np.isnan(se):
        return mean, mean
    t_val = stats.t.ppf(1 - alpha / 2, df=len(values) - 1)
    return mean - t_val * se, mean + t_val * se


def permutation_test(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    num_permutations: int = 1000,
    random_state: int | None = None,
) -> float:
    rng = np.random.default_rng(random_state)
    observed = metric_fn(y_true, y_score)
    count = 0
    for _ in range(num_permutations):
        permuted = rng.permutation(y_true)
        score = metric_fn(permuted, y_score)
        if score >= observed:
            count += 1
    return (count + 1) / (num_permutations + 1)


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(p_values)
    ranked = np.arange(1, len(p_values) + 1)
    thresholds = alpha * ranked / len(p_values)
    accepted = p_values[order] <= thresholds
    return accepted, thresholds


__all__ = [
    "macro_f1",
    "macro_auroc",
    "confidence_interval",
    "permutation_test",
    "fdr_correction",
]
