"""Gradient checks, symmetries, and a tiny shape fit."""

import numpy as np

from wsp.controls import canonicalize, permute_neurons, shuffle_within_tensors
from wsp.diffusion import Denoiser, cosine_schedule
from wsp.mlp import (
    bce_with_logits,
    flatten_weights,
    forward_logits,
    init_ensemble,
    loss_and_grads,
    predict_proba,
    train_ensemble,
    unflatten_weights,
)
from wsp.shapes import grid_coords, rasterize, sample_specs


def test_flatten_roundtrip():
    rng = np.random.default_rng(0)
    ens = init_ensemble(3, 2, 5, rng)
    flat = flatten_weights(ens)
    back = unflatten_weights(flat, 2, 5)
    for left, right in zip(ens.arrays(), back.arrays()):
        np.testing.assert_allclose(left, right)


def test_bce_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    ens = init_ensemble(1, 2, 4, rng)
    coords = rng.normal(size=(7, 2))
    target = (rng.random(7) > 0.5).astype(np.float64)[None, :]
    _, grads = loss_and_grads(ens, coords, target)
    eps = 1e-6
    # One weight in each tensor.
    checks = [
        (0, (0, 0, 0)),
        (1, (0, 1)),
        (2, (0, 2, 1)),
        (4, (0, 0, 3)),
    ]
    for array_index, where in checks:
        array = ens.arrays()[array_index]
        original = array[where]
        array[where] = original + eps
        plus = bce_with_logits(forward_logits(ens, coords)[0], target)
        array[where] = original - eps
        minus = bce_with_logits(forward_logits(ens, coords)[0], target)
        array[where] = original
        numeric = (plus - minus) / (2 * eps)
        assert abs(numeric - grads[array_index][where]) < 1e-5


def test_permutation_preserves_function_and_shuffle_does_not():
    rng = np.random.default_rng(2)
    ens = init_ensemble(4, 2, 6, rng)
    coords = rng.normal(size=(20, 2))
    original = predict_proba(ens, coords)
    permuted = predict_proba(permute_neurons(ens, rng), coords)
    np.testing.assert_allclose(original, permuted, atol=1e-8)
    shuffled = predict_proba(shuffle_within_tensors(ens, np.random.default_rng(3)), coords)
    assert np.mean(np.abs(original - shuffled)) > 1e-3


def test_canonicalize_removes_permutation():
    rng = np.random.default_rng(4)
    ens = init_ensemble(5, 3, 7, rng)
    canon = canonicalize(ens)
    again = canonicalize(permute_neurons(ens, rng))
    np.testing.assert_allclose(flatten_weights(canon), flatten_weights(again), atol=1e-8)
    np.testing.assert_allclose(flatten_weights(canon), flatten_weights(canonicalize(canon)))


def test_circle_and_rotated_square_fit():
    rng = np.random.default_rng(5)
    coords = grid_coords(16, 2)
    specs = sample_specs(["circle", "square"], 1, rng, 2, 0.1, 0.4, 0.5)
    target = rasterize(specs, coords)
    ens = init_ensemble(2, 2, 12, np.random.default_rng(6))
    train_ensemble(ens, coords, target, steps=180, lr=0.04)
    pred = predict_proba(ens, coords) >= 0.5
    inter = np.logical_and(pred, target >= 0.5).sum(axis=1)
    union = np.logical_or(pred, target >= 0.5).sum(axis=1)
    iou = inter / np.maximum(union, 1)
    assert iou[0] > 0.85
    assert iou[1] > 0.75


def test_denoiser_gradient_and_schedule():
    rng = np.random.default_rng(7)
    model = Denoiser.create(5, 4, 3, rng)
    x = rng.normal(size=(6, 5))
    pred, cache = model.forward(x)
    dout = np.ones_like(pred) / pred.size
    grads = model.grads(cache, dout)
    eps = 1e-6
    original = model.w1[0, 0]
    model.w1[0, 0] = original + eps
    plus = model.forward(x)[0].sum() / pred.size
    model.w1[0, 0] = original - eps
    minus = model.forward(x)[0].sum() / pred.size
    model.w1[0, 0] = original
    numeric = (plus - minus) / (2 * eps)
    assert abs(numeric - grads[0][0, 0]) < 1e-5
    schedule = cosine_schedule(10)
    assert np.all(np.diff(schedule["alpha_bars"]) < 0)
    assert schedule["alpha_bars"][0] < 1.0
