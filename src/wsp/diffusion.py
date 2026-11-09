"""A small epsilon-predicting DDPM over flattened network weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_schedule(timesteps: int, s: float = 0.008) -> dict[str, np.ndarray]:
    steps = np.arange(timesteps + 1, dtype=np.float64)
    f = np.cos(((steps / timesteps) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = np.clip(1.0 - (alpha_bar[1:] / alpha_bar[:-1]), 1e-5, 0.999)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}


def linear_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.05) -> dict[str, np.ndarray]:
    """Ho et al. linear schedule. beta_end stays moderate so ancestral steps cannot explode."""

    betas = np.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}


def timestep_embedding(timesteps: np.ndarray, dim: int) -> np.ndarray:
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    args = timesteps.astype(np.float64)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)


def _silu(z: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    return z * sig


def _silu_grad(z: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    return sig * (1.0 + z * (1.0 - sig))


@dataclass
class Denoiser:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    @classmethod
    def create(cls, in_dim: int, hidden: int, out_dim: int, rng: np.random.Generator) -> Denoiser:
        def weight(fan_in: int, fan_out: int) -> np.ndarray:
            bound = np.sqrt(6.0 / fan_in)
            return rng.uniform(-bound, bound, size=(fan_out, fan_in)).astype(np.float64)

        return cls(
            weight(in_dim, hidden),
            np.zeros(hidden, dtype=np.float64),
            weight(hidden, hidden),
            np.zeros(hidden, dtype=np.float64),
            weight(hidden, out_dim),
            np.zeros(out_dim, dtype=np.float64),
        )

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        z1 = x @ self.w1.T + self.b1
        h1 = _silu(z1)
        z2 = h1 @ self.w2.T + self.b2
        h2 = _silu(z2)
        out = h2 @ self.w3.T + self.b3
        return out, (x, z1, h1, z2, h2)

    def grads(self, cache: tuple[np.ndarray, ...], dout: np.ndarray) -> tuple[np.ndarray, ...]:
        x, z1, h1, z2, h2 = cache
        dw3 = dout.T @ h2
        db3 = dout.sum(axis=0)
        dh2 = dout @ self.w3
        dz2 = dh2 * _silu_grad(z2)
        dw2 = dz2.T @ h1
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.w2
        dz1 = dh1 * _silu_grad(z1)
        dw1 = dz1.T @ x
        db1 = dz1.sum(axis=0)
        return (dw1, db1, dw2, db2, dw3, db3)


def _adam_step(
    arrays: tuple[np.ndarray, ...],
    grads: tuple[np.ndarray, ...],
    state: dict[str, list],
    lr: float,
) -> None:
    state["t"] += 1
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for index, (array, grad) in enumerate(zip(arrays, grads)):
        state["m"][index] = beta1 * state["m"][index] + (1.0 - beta1) * grad
        state["v"][index] = beta2 * state["v"][index] + (1.0 - beta2) * (grad * grad)
        mhat = state["m"][index] / (1.0 - beta1 ** state["t"])
        vhat = state["v"][index] / (1.0 - beta2 ** state["t"])
        array -= lr * mhat / (np.sqrt(vhat) + eps)


def _features(
    noisy: np.ndarray,
    timesteps: np.ndarray,
    labels: np.ndarray | None,
    n_classes: int,
    time_dim: int,
) -> np.ndarray:
    parts = [noisy, timestep_embedding(timesteps, time_dim)]
    if labels is not None:
        one_hot = np.zeros((noisy.shape[0], n_classes), dtype=np.float64)
        one_hot[np.arange(noisy.shape[0]), labels] = 1.0
        parts.append(one_hot)
    return np.concatenate(parts, axis=1)


def train_denoiser(
    x0: np.ndarray,
    labels: np.ndarray | None,
    schedule: dict[str, np.ndarray],
    steps: int,
    batch_size: int,
    hidden: int,
    lr: float,
    time_dim: int,
    n_classes: int,
    rng: np.random.Generator,
) -> Denoiser:
    n, dim = x0.shape
    in_dim = dim + time_dim + (n_classes if labels is not None else 0)
    model = Denoiser.create(in_dim, hidden, dim, rng)
    state = {"t": 0, "m": [np.zeros_like(a) for a in model.arrays()], "v": [np.zeros_like(a) for a in model.arrays()]}
    alpha_bars = schedule["alpha_bars"]
    n_steps = alpha_bars.shape[0]
    for _ in range(steps):
        choice = rng.integers(0, n, size=batch_size)
        batch = x0[choice]
        batch_labels = None if labels is None else labels[choice]
        t = rng.integers(0, n_steps, size=batch_size)
        eps = rng.normal(size=batch.shape)
        scale = np.sqrt(alpha_bars[t])[:, None]
        sigma = np.sqrt(1.0 - alpha_bars[t])[:, None]
        noisy = scale * batch + sigma * eps
        features = _features(noisy, t, batch_labels, n_classes, time_dim)
        pred, cache = model.forward(features)
        # Mean squared error over the batch and the coordinates.
        dout = 2.0 * (pred - eps) / pred.size
        grads = model.grads(cache, dout)
        _adam_step(model.arrays(), grads, state, lr)
    return model


def predict_eps(
    model: Denoiser,
    noisy: np.ndarray,
    timesteps: np.ndarray,
    labels: np.ndarray | None,
    n_classes: int,
    time_dim: int,
) -> np.ndarray:
    features = _features(noisy, timesteps, labels, n_classes, time_dim)
    pred, _ = model.forward(features)
    return pred


def denoising_se(
    model: Denoiser,
    x0: np.ndarray,
    labels: np.ndarray | None,
    schedule: dict[str, np.ndarray],
    timestep: int,
    time_dim: int,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-example mean squared error of the noise prediction."""

    t = np.full(x0.shape[0], timestep, dtype=np.int64)
    eps = rng.normal(size=x0.shape)
    alpha_bar = schedule["alpha_bars"][timestep]
    noisy = np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * eps
    pred = predict_eps(model, noisy, t, labels, n_classes, time_dim)
    return np.mean((pred - eps) ** 2, axis=1)


def denoising_mse(
    model: Denoiser,
    x0: np.ndarray,
    labels: np.ndarray | None,
    schedule: dict[str, np.ndarray],
    timestep: int,
    time_dim: int,
    n_classes: int,
    rng: np.random.Generator,
) -> float:
    return float(np.mean(denoising_se(model, x0, labels, schedule, timestep, time_dim, n_classes, rng)))


def sample(
    model: Denoiser,
    n: int,
    labels: np.ndarray | None,
    schedule: dict[str, np.ndarray],
    time_dim: int,
    n_classes: int,
    rng: np.random.Generator,
    clip: float = 8.0,
) -> np.ndarray:
    """Ancestral sample. Predicted clean weights are clipped, which keeps a bad step finite.

    Standardized trained weights in this experiment sit well inside [-8, 8].
    """

    dim = model.w3.shape[0]
    x = rng.normal(size=(n, dim))
    alphas = schedule["alphas"]
    alpha_bars = schedule["alpha_bars"]
    betas = schedule["betas"]
    for t in range(len(alphas) - 1, -1, -1):
        timesteps = np.full(n, t, dtype=np.int64)
        eps = predict_eps(model, x, timesteps, labels, n_classes, time_dim)
        alpha_bar = alpha_bars[t]
        x0 = (x - np.sqrt(1.0 - alpha_bar) * eps) / np.sqrt(max(alpha_bar, 1e-12))
        x0 = np.clip(x0, -clip, clip)
        if t == 0:
            return x0
        alpha_prev = alpha_bars[t - 1]
        coef_x0 = np.sqrt(alpha_prev) * betas[t] / (1.0 - alpha_bar)
        coef_xt = np.sqrt(alphas[t]) * (1.0 - alpha_prev) / (1.0 - alpha_bar)
        mean = coef_x0 * x0 + coef_xt * x
        var = betas[t] * (1.0 - alpha_prev) / (1.0 - alpha_bar)
        x = mean + np.sqrt(max(var, 0.0)) * rng.normal(size=x.shape)
    return x
