"""IoU, template classification, ridge probe, nearest centroid, calibration."""

from __future__ import annotations

import numpy as np


def batched_iou(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    inter = np.logical_and(pred_b, target_b).sum(axis=-1)
    union = np.logical_or(pred_b, target_b).sum(axis=-1)
    return inter / np.maximum(union, 1)


def best_template(
    renders: np.ndarray,
    bank: np.ndarray,
    bank_labels: list[str],
    threshold: float,
) -> tuple[list[str], np.ndarray]:
    """renders and bank are boolean arrays (N, G) and (M, G)."""

    render_f = renders.astype(np.float64)
    bank_f = bank.astype(np.float64)
    inter = render_f @ bank_f.T
    union = render_f.sum(axis=1, keepdims=True) + bank_f.sum(axis=1)[None, :] - inter
    ious = inter / np.maximum(union, 1.0)
    best = np.argmax(ious, axis=1)
    best_iou = ious[np.arange(renders.shape[0]), best]
    labels = [bank_labels[int(i)] if float(score) >= threshold else "unrecognized" for i, score in zip(best, best_iou)]
    return labels, best_iou


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def _one_hot(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(classes)}
    out = np.zeros((labels.shape[0], len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        out[row, index[str(label)]] = 1.0
    return out


def fit_ridge(x: np.ndarray, labels: np.ndarray, classes: list[str], l2: float) -> np.ndarray:
    """Ridge regression onto one-hot labels, with an intercept column. Returns weights (d+1, C)."""

    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    target = _one_hot(labels, classes)
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += l2
    # Do not regularize the intercept as strongly: undo the last diagonal add and use a small one.
    gram[-1, -1] -= l2
    gram[-1, -1] += 1e-6
    return np.linalg.solve(gram, design.T @ target)


def predict_ridge(x: np.ndarray, weight: np.ndarray, classes: list[str]) -> list[str]:
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    scores = design @ weight
    chosen = np.argmax(scores, axis=1)
    return [classes[int(i)] for i in chosen]


def accuracy(pred: list[str], truth: list[str] | np.ndarray) -> float:
    truth_list = [str(v) for v in truth]
    if not truth_list:
        return float("nan")
    hits = sum(p == t for p, t in zip(pred, truth_list))
    return hits / len(truth_list)


def fit_centroids(x: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict[str, np.ndarray]:
    return {name: x[labels == name].mean(axis=0) for name in classes}


def centroid_distances(x: np.ndarray, centroids: dict[str, np.ndarray], classes: list[str]) -> np.ndarray:
    return np.stack([np.linalg.norm(x - centroids[name], axis=1) for name in classes], axis=1)


def predict_centroid(
    x: np.ndarray,
    centroids: dict[str, np.ndarray],
    classes: list[str],
    reject_distance: float | None = None,
) -> list[str]:
    dists = centroid_distances(x, centroids, classes)
    chosen = np.argmin(dists, axis=1)
    labels = []
    for row, index in enumerate(chosen):
        if reject_distance is not None and float(dists[row, index]) > reject_distance:
            labels.append("unrecognized")
        else:
            labels.append(classes[int(index)])
    return labels


def reject_threshold(x: np.ndarray, labels: np.ndarray, centroids: dict[str, np.ndarray], classes: list[str]) -> float:
    """95th percentile of training distance to the point's own centroid. Fixed before test time."""

    own = []
    index = {name: i for i, name in enumerate(classes)}
    dists = centroid_distances(x, centroids, classes)
    for row, label in enumerate(labels):
        own.append(dists[row, index[str(label)]])
    return float(np.quantile(np.asarray(own), 0.95))


def confusion(pred: list[str], truth: list[str], labels: list[str]) -> list[list[int]]:
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for guess, true in zip(pred, truth):
        if guess not in index or true not in index:
            continue
        matrix[index[true], index[guess]] += 1
    return matrix.tolist()


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Confidence Brier against correctness. The model reports one probability, not a full distribution."""

    if len(confidence) == 0:
        return float("nan")
    return float(np.mean((confidence - correct.astype(np.float64)) ** 2))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 5) -> dict:
    if len(confidence) == 0:
        return {"ece": float("nan"), "bins": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": 0, "accuracy": None, "confidence": None})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        ece += (count / total) * abs(acc - conf)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": count, "accuracy": acc, "confidence": conf})
    return {"ece": float(ece), "bins": bins}


def class_counts(labels: list[str], classes: list[str]) -> dict[str, int]:
    return {name: sum(label == name for label in labels) for name in classes}
