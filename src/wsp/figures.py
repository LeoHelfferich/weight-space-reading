"""Aggregate figures. Numbers live in the summary; these plots are the pictures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wsp.mlp import Ensemble

COLORS = {
    "circle": "#0072B2",
    "square": "#E69F00",
    "triangle": "#009E73",
    "sphere": "#0072B2",
    "cube": "#E69F00",
    "octahedron": "#009E73",
    "unrecognized": "#999999",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _as_image(mask: np.ndarray, resolution: int, dim: int) -> np.ndarray:
    if dim == 2:
        return mask.reshape(resolution, resolution)
    volume = mask.reshape(resolution, resolution, resolution)
    return volume[resolution // 2]


def save_mask_grid(
    path: Path,
    rows: list[tuple[str, np.ndarray]],
    resolution: int,
    dim: int,
    title: str,
) -> None:
    n = rows[0][1].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.35 * n, 1.45 * len(rows)), squeeze=False)
    for r, (label, masks) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(_as_image(masks[c], resolution, dim), cmap="gray_r", vmin=0, vmax=1, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(title, fontsize=12)
    _save(fig, path)


def save_pca(path: Path, panels: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.0), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        xy = panel["xy"]
        labels = panel["labels"]
        for name in panel["classes"]:
            mask = np.array([label == name for label in labels])
            ax.scatter(xy[mask, 0], xy[mask, 1], s=18, c=COLORS.get(name, "#333333"), label=name, alpha=0.85)
        if "extra_xy" in panel:
            ax.scatter(
                panel["extra_xy"][:, 0],
                panel["extra_xy"][:, 1],
                s=14,
                c="#999999",
                marker="x",
                label=panel.get("extra_label", "shuffled"),
                alpha=0.7,
            )
        ax.set_title(panel["title"])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def save_bars(path: Path, title: str, groups: list[str], series: dict[str, list[float]], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(groups))
    width = 0.8 / max(len(series), 1)
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width=width, label=name, color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(1.0 / 3.0, color="#666666", lw=0.8, ls="--", label="ridge chance 1/3")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def save_hyperplanes(
    path: Path,
    coords: np.ndarray,
    masks: list[np.ndarray],
    networks: list[Ensemble],
    titles: list[str],
    resolution: int,
) -> None:
    fig, axes = plt.subplots(1, len(masks), figsize=(4.0 * len(masks), 3.8), squeeze=False)
    for ax, mask, ens, title in zip(axes[0], masks, networks, titles):
        image = mask.reshape(resolution, resolution)
        ax.imshow(
            image,
            cmap="gray_r",
            vmin=0,
            vmax=1,
            origin="lower",
            extent=(-1, 1, -1, 1),
        )
        influence = np.abs(ens.w3[0, 0]) @ np.abs(ens.w2[0])
        influence = influence / (float(influence.max()) + 1e-8)
        for unit in range(ens.hidden):
            _draw_line(ax, ens.w1[0, unit], float(ens.b1[0, unit]), float(influence[unit]))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    _save(fig, path)


def _draw_line(ax: plt.Axes, weight: np.ndarray, bias: float, importance: float) -> None:
    norm = float(np.linalg.norm(weight))
    if norm < 1e-6:
        return
    point = -bias * weight / (norm**2)
    direction = np.array([-weight[1], weight[0]], dtype=np.float64) / norm
    t = np.linspace(-2.5, 2.5, 80)
    pts = point[None, :] + t[:, None] * direction[None, :]
    keep = np.all(np.abs(pts) <= 1.02, axis=1)
    if int(keep.sum()) < 2:
        return
    ax.plot(pts[keep, 0], pts[keep, 1], color="#D55E00", lw=0.4 + 2.2 * importance, alpha=0.35 + 0.55 * importance)
