"""Synthetic filled shapes on a regular grid. No external datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShapeSpec:
    """One shape instance. Rotation is a radians angle in 2D, or a unit quaternion in 3D."""

    name: str
    center: tuple[float, ...]
    scale: float
    rotation: tuple[float, ...]


def grid_coords(resolution: int, dim: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, resolution)
    mesh = np.meshgrid(*([axis] * dim), indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1)


def sample_specs(
    names: list[str],
    n_per_class: int,
    rng: np.random.Generator,
    dim: int,
    center_limit: float,
    scale_min: float,
    scale_max: float,
) -> list[ShapeSpec]:
    specs: list[ShapeSpec] = []
    for name in names:
        for _ in range(n_per_class):
            center = tuple(float(v) for v in rng.uniform(-center_limit, center_limit, size=dim))
            scale = float(rng.uniform(scale_min, scale_max))
            if dim == 2:
                rotation = (float(rng.uniform(0.0, 2.0 * np.pi)),)
            elif dim == 3:
                quat = rng.normal(size=4)
                quat = quat / np.linalg.norm(quat)
                rotation = tuple(float(v) for v in quat)
            else:
                raise ValueError(f"unsupported dim {dim}")
            specs.append(ShapeSpec(name, center, scale, rotation))
    return specs


def _quat_to_matrix(quat: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _into_shape_frame(points: np.ndarray, spec: ShapeSpec) -> np.ndarray:
    centered = points - np.asarray(spec.center, dtype=np.float64)
    if points.shape[1] == 2:
        theta = spec.rotation[0]
        cosine, sine = np.cos(theta), np.sin(theta)
        rot = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
        return centered @ rot.T
    rot = _quat_to_matrix(spec.rotation)
    return centered @ rot.T


def _inside(local: np.ndarray, spec: ShapeSpec) -> np.ndarray:
    scale = spec.scale
    if spec.name in {"circle", "sphere"}:
        return np.sum(local * local, axis=1) <= scale * scale
    if spec.name in {"square", "cube"}:
        return np.max(np.abs(local), axis=1) <= scale
    if spec.name == "triangle":
        # Equilateral triangle, circumradius = scale, one vertex up after rotation.
        theta = spec.rotation[0]
        angles = theta + np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])
        verts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * scale
        return _point_in_triangle(local, verts)
    if spec.name == "octahedron":
        return np.sum(np.abs(local), axis=1) <= scale
    raise ValueError(f"unknown shape {spec.name}")


def _point_in_triangle(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts
    den = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    a = ((v1[1] - v2[1]) * (points[:, 0] - v2[0]) + (v2[0] - v1[0]) * (points[:, 1] - v2[1])) / den
    b = ((v2[1] - v0[1]) * (points[:, 0] - v2[0]) + (v0[0] - v2[0]) * (points[:, 1] - v2[1])) / den
    c = 1.0 - a - b
    return (a >= 0.0) & (b >= 0.0) & (c >= 0.0)


def rasterize(specs: list[ShapeSpec], coords: np.ndarray) -> np.ndarray:
    masks = np.stack([_inside(_into_shape_frame(coords, spec), spec) for spec in specs], axis=0)
    return masks.astype(np.float64)


def template_bank(
    dim: int,
    resolution: int,
    names: list[str],
    center_limit: float,
    scale_min: float,
    scale_max: float,
) -> tuple[np.ndarray, list[str]]:
    """Coarse pose grid used to score renders that have no ground-truth mask."""

    coords = grid_coords(resolution, dim)
    centers_1d = np.linspace(-center_limit, center_limit, 3)
    scales = np.linspace(scale_min, scale_max, 3)
    if dim == 2:
        angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        rotations = [(float(a),) for a in angles]
    else:
        rotations = [
            (1.0, 0.0, 0.0, 0.0),
            (0.70710678, 0.70710678, 0.0, 0.0),
            (0.70710678, 0.0, 0.70710678, 0.0),
            (0.70710678, 0.0, 0.0, 0.70710678),
        ]
    specs: list[ShapeSpec] = []
    for name in names:
        use_rotations = rotations
        if name in {"circle", "sphere"}:
            use_rotations = [rotations[0]]
        for center_vals in np.array(np.meshgrid(*([centers_1d] * dim), indexing="ij")).reshape(dim, -1).T:
            center = tuple(float(v) for v in center_vals)
            for scale in scales:
                for rotation in use_rotations:
                    specs.append(ShapeSpec(name, center, float(scale), rotation))
    masks = rasterize(specs, coords) >= 0.5
    labels = [spec.name for spec in specs]
    return masks, labels
