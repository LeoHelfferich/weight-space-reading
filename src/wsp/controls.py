"""Weight-space controls: neuron permutation, within-tensor shuffle, canonicalization."""

from __future__ import annotations

import numpy as np

from wsp.mlp import Ensemble, flatten_weights, init_ensemble


def apply_hidden_perms(ens: Ensemble, perm1: np.ndarray, perm2: np.ndarray) -> Ensemble:
    """Permute hidden units. perm[n, new] = old index. ReLU MLPs are invariant to this."""

    out = ens.copy()
    for n in range(ens.n):
        first = perm1[n]
        second = perm2[n]
        out.w1[n] = ens.w1[n, first]
        out.b1[n] = ens.b1[n, first]
        out.w2[n] = ens.w2[n, second][:, first]
        out.b2[n] = ens.b2[n, second]
        out.w3[n] = ens.w3[n][:, second]
    return out


def permute_neurons(ens: Ensemble, rng: np.random.Generator) -> Ensemble:
    hidden = ens.hidden
    perm1 = np.stack([rng.permutation(hidden) for _ in range(ens.n)], axis=0)
    perm2 = np.stack([rng.permutation(hidden) for _ in range(ens.n)], axis=0)
    return apply_hidden_perms(ens, perm1, perm2)


def shuffle_within_tensors(ens: Ensemble, rng: np.random.Generator) -> Ensemble:
    """Shuffle entries inside each tensor. Layer scales stay, geometry does not."""

    out = ens.copy()
    for array in out.arrays():
        flat = array.reshape(array.shape[0], -1)
        for n in range(flat.shape[0]):
            flat[n] = rng.permutation(flat[n])
    return out


def random_like(ens: Ensemble, rng: np.random.Generator) -> Ensemble:
    return init_ensemble(ens.n, ens.in_dim, ens.hidden, rng, out_dim=ens.out_dim, shared_init=False)


def _unit_order(weight_rows: np.ndarray) -> np.ndarray:
    score = np.linalg.norm(weight_rows, axis=1)
    tie = weight_rows.sum(axis=1)
    return np.lexsort((tie, score))


def canonicalize(ens: Ensemble) -> Ensemble:
    """Sort hidden units by incoming-weight norm. A cheap gauge fix, not a full alignment."""

    out = ens.copy()
    hidden = ens.hidden
    for n in range(ens.n):
        first = _unit_order(out.w1[n])
        out.w1[n] = out.w1[n, first]
        out.b1[n] = out.b1[n, first]
        out.w2[n] = out.w2[n][:, first]
        second = _unit_order(out.w2[n])
        out.w2[n] = out.w2[n, second]
        out.b2[n] = out.b2[n, second]
        out.w3[n] = out.w3[n][:, second]
        if first.shape != (hidden,) or second.shape != (hidden,):
            raise RuntimeError("bad permutation shape")
    return out


def layer_slices(in_dim: int, hidden: int, out_dim: int = 1) -> dict[str, slice]:
    """Column ranges in the flattened layout, matching flatten_weights."""

    cuts = [
        ("w1", hidden * in_dim),
        ("b1", hidden),
        ("w2", hidden * hidden),
        ("b2", hidden),
        ("w3", out_dim * hidden),
        ("b3", out_dim),
    ]
    slices: dict[str, slice] = {}
    start = 0
    for name, width in cuts:
        slices[name] = slice(start, start + width)
        start += width
    return slices


def rounded_flat(ens: Ensemble, decimals: int) -> np.ndarray:
    flat = np.round(flatten_weights(ens), decimals)
    flat[flat == 0.0] = 0.0
    return flat
