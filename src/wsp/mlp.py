"""Small ReLU MLPs, including a vectorized ensemble trainer for implicit shapes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAM_NAMES = ("w1", "b1", "w2", "b2", "w3", "b3")


def parameter_count(in_dim: int, hidden: int, out_dim: int = 1) -> int:
    return hidden * in_dim + hidden + hidden * hidden + hidden + out_dim * hidden + out_dim


def _kaiming_uniform(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    bound = np.sqrt(6.0 / fan_in)
    return rng.uniform(-bound, bound, size=shape).astype(np.float64)


@dataclass
class Ensemble:
    """N networks with layout Linear(in, H) - ReLU - Linear(H, H) - ReLU - Linear(H, out)."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    def copy(self) -> Ensemble:
        return Ensemble(*(array.copy() for array in self.arrays()))

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)

    @property
    def n(self) -> int:
        return int(self.w1.shape[0])

    @property
    def hidden(self) -> int:
        return int(self.w1.shape[1])

    @property
    def in_dim(self) -> int:
        return int(self.w1.shape[2])

    @property
    def out_dim(self) -> int:
        return int(self.w3.shape[1])


def init_ensemble(
    n: int,
    in_dim: int,
    hidden: int,
    rng: np.random.Generator,
    out_dim: int = 1,
    shared_init: bool = False,
) -> Ensemble:
    if shared_init:
        one = init_ensemble(1, in_dim, hidden, rng, out_dim=out_dim, shared_init=False)
        return Ensemble(
            np.repeat(one.w1, n, axis=0),
            np.repeat(one.b1, n, axis=0),
            np.repeat(one.w2, n, axis=0),
            np.repeat(one.b2, n, axis=0),
            np.repeat(one.w3, n, axis=0),
            np.repeat(one.b3, n, axis=0),
        )
    return Ensemble(
        _kaiming_uniform(rng, (n, hidden, in_dim), in_dim),
        np.zeros((n, hidden), dtype=np.float64),
        _kaiming_uniform(rng, (n, hidden, hidden), hidden),
        np.zeros((n, hidden), dtype=np.float64),
        _kaiming_uniform(rng, (n, out_dim, hidden), hidden),
        np.zeros((n, out_dim), dtype=np.float64),
    )


def flatten_weights(ens: Ensemble) -> np.ndarray:
    parts = [array.reshape(ens.n, -1) for array in ens.arrays()]
    return np.concatenate(parts, axis=1)


def unflatten_weights(flat: np.ndarray, in_dim: int, hidden: int, out_dim: int = 1) -> Ensemble:
    expected = parameter_count(in_dim, hidden, out_dim)
    if flat.ndim != 2 or flat.shape[1] != expected:
        raise ValueError(f"expected (N, {expected}) weights, got {flat.shape}")
    n = flat.shape[0]
    index = 0

    def take(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal index
        count = int(np.prod(shape))
        chunk = flat[:, index : index + count].reshape((n, *shape))
        index += count
        return chunk

    return Ensemble(
        take((hidden, in_dim)),
        take((hidden,)),
        take((hidden, hidden)),
        take((hidden,)),
        take((out_dim, hidden)),
        take((out_dim,)),
    )


def forward_logits(ens: Ensemble, coords: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    z1 = np.einsum("gi,nhi->ngh", coords, ens.w1, optimize=True) + ens.b1[:, None, :]
    h1 = np.maximum(z1, 0.0)
    z2 = np.einsum("ngi,nji->ngj", h1, ens.w2, optimize=True) + ens.b2[:, None, :]
    h2 = np.maximum(z2, 0.0)
    logits = np.einsum("ngi,noi->ngo", h2, ens.w3, optimize=True) + ens.b3[:, None, :]
    return logits, (z1, h1, z2, h2)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def predict_proba(ens: Ensemble, coords: np.ndarray) -> np.ndarray:
    logits, _ = forward_logits(ens, coords)
    return sigmoid(logits[..., 0])


def bce_with_logits(logits: np.ndarray, target: np.ndarray) -> float:
    z = np.clip(logits[..., 0], -40.0, 40.0)
    loss = np.maximum(z, 0.0) - z * target + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(loss))


def loss_and_grads(
    ens: Ensemble, coords: np.ndarray, target: np.ndarray
) -> tuple[float, tuple[np.ndarray, ...]]:
    """Mean BCE over pixels. Gradients are per network and are not divided by N."""

    logits, (z1, h1, z2, h2) = forward_logits(ens, coords)
    loss = bce_with_logits(logits, target)
    n_points = target.shape[1]
    residual = (sigmoid(logits) - target[:, :, None]) / n_points
    dw3 = np.einsum("ngo,ngi->noi", residual, h2, optimize=True)
    db3 = residual.sum(axis=1)
    dh2 = np.einsum("ngo,noi->ngi", residual, ens.w3, optimize=True)
    dz2 = dh2 * (z2 > 0.0)
    dw2 = np.einsum("ngj,ngi->nji", dz2, h1, optimize=True)
    db2 = dz2.sum(axis=1)
    dh1 = np.einsum("ngj,nji->ngi", dz2, ens.w2, optimize=True)
    dz1 = dh1 * (z1 > 0.0)
    dw1 = np.einsum("ngh,gi->nhi", dz1, coords, optimize=True)
    db1 = dz1.sum(axis=1)
    return loss, (dw1, db1, dw2, db2, dw3, db3)


class Adam:
    def __init__(self, arrays: tuple[np.ndarray, ...], lr: float):
        self.lr = lr
        self.m = [np.zeros_like(array) for array in arrays]
        self.v = [np.zeros_like(array) for array in arrays]
        self.t = 0

    def step(self, arrays: tuple[np.ndarray, ...], grads: tuple[np.ndarray, ...]) -> None:
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for index, (array, grad) in enumerate(zip(arrays, grads)):
            self.m[index] = beta1 * self.m[index] + (1.0 - beta1) * grad
            self.v[index] = beta2 * self.v[index] + (1.0 - beta2) * (grad * grad)
            mhat = self.m[index] / (1.0 - beta1**self.t)
            vhat = self.v[index] / (1.0 - beta2**self.t)
            array -= self.lr * mhat / (np.sqrt(vhat) + eps)


def train_ensemble(
    ens: Ensemble,
    coords: np.ndarray,
    target: np.ndarray,
    steps: int,
    lr: float,
) -> list[float]:
    opt = Adam(ens.arrays(), lr)
    losses: list[float] = []
    for _ in range(steps):
        loss, grads = loss_and_grads(ens, coords, target)
        opt.step(ens.arrays(), grads)
        losses.append(loss)
    return losses
