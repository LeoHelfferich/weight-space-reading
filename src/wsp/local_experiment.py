"""Train shape INRs and a small DDPM over their weights.

Run: python -m wsp.local_experiment --config configs/local.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from wsp import __version__
from wsp.controls import canonicalize, layer_slices, permute_neurons, random_like, shuffle_within_tensors
from wsp.diffusion import denoising_se, linear_schedule, sample, train_denoiser
from wsp.figures import save_bars, save_hyperplanes, save_mask_grid, save_pca
from wsp.jsonutil import to_jsonable
from wsp.metrics import (
    accuracy,
    apply_standardizer,
    batched_iou,
    best_template,
    class_counts,
    fit_centroids,
    fit_ridge,
    fit_standardizer,
    predict_centroid,
    predict_ridge,
)
from wsp.mlp import (
    Ensemble,
    flatten_weights,
    init_ensemble,
    parameter_count,
    predict_proba,
    train_ensemble,
    unflatten_weights,
)
from wsp.shapes import grid_coords, rasterize, sample_specs, template_bank


def mix_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _labels(names: list[str], n_per: int) -> np.ndarray:
    return np.repeat(np.array(names, dtype=object), n_per)


def _subset(ens: Ensemble, index: np.ndarray) -> Ensemble:
    return Ensemble(*(array[index] for array in ens.arrays()))


def _one(ens: Ensemble, index: int) -> Ensemble:
    return _subset(ens, np.array([index]))


def _iou_to_target(ens: Ensemble, target: np.ndarray, coords: np.ndarray) -> np.ndarray:
    pred = predict_proba(ens, coords) >= 0.5
    return batched_iou(pred, target >= 0.5)


def _per_class(values: np.ndarray, names: list[str], n_per: int) -> dict[str, float]:
    return {name: float(values[i * n_per : (i + 1) * n_per].mean()) for i, name in enumerate(names)}


def _pca(x: np.ndarray, k: int) -> dict[str, np.ndarray | int]:
    mu = x.mean(axis=0)
    centered = x - mu
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    k = int(min(k, vt.shape[0], max(x.shape[0] - 1, 1)))
    variance = singular**2
    variance = variance / max(float(variance.sum()), 1e-12)
    return {"mu": mu, "basis": vt[:k], "k": k, "explained": float(variance[:k].sum()), "components": vt[:2]}


def _to_latent(x: np.ndarray, pca: dict) -> np.ndarray:
    return (x - pca["mu"]) @ pca["basis"].T


def _from_latent(z: np.ndarray, pca: dict) -> np.ndarray:
    return z @ pca["basis"] + pca["mu"]


def _fit_latent_diffusion(x: np.ndarray, labels: np.ndarray, cfg: dict, rng: np.random.Generator, k: int):
    pca = _pca(x, k)
    latent = _to_latent(x, pca)
    mu, sd = fit_standardizer(latent)
    standardized = apply_standardizer(latent, mu, sd)
    names = cfg["class_names"]
    label_index = np.array([names.index(str(label)) for label in labels], dtype=np.int64)
    schedule = linear_schedule(int(cfg["timesteps"]), beta_end=float(cfg["beta_end"]))
    model = train_denoiser(
        standardized,
        label_index,
        schedule,
        steps=int(cfg["train_steps"]),
        batch_size=int(min(cfg["batch_size"], standardized.shape[0])),
        hidden=int(cfg["hidden"]),
        lr=float(cfg["lr"]),
        time_dim=int(cfg["time_dim"]),
        n_classes=len(names),
        rng=rng,
    )
    pack = {
        "pca": pca,
        "mu": mu,
        "sd": sd,
        "model": model,
        "schedule": schedule,
        "standardized": standardized,
        "label_index": label_index,
    }
    return pack


def _encode(flat: np.ndarray, pack: dict) -> np.ndarray:
    latent = _to_latent(flat, pack["pca"])
    return apply_standardizer(latent, pack["mu"], pack["sd"])


def _decode(standardized: np.ndarray, pack: dict, weight_mu: np.ndarray, weight_sd: np.ndarray) -> np.ndarray:
    latent = standardized * pack["sd"] + pack["mu"]
    flat = _from_latent(latent, pack["pca"])
    return flat * weight_sd + weight_mu


def _score_renders(flat: np.ndarray, requested: list[str] | None, in_dim: int, hidden: int, coords, bank, bank_labels, threshold: float) -> dict:
    ens = unflatten_weights(flat, in_dim, hidden)
    renders = predict_proba(ens, coords) >= 0.5
    hat, best = best_template(renders, bank, bank_labels, threshold)
    known = list(dict.fromkeys(bank_labels)) + ["unrecognized"]
    out = {
        "n": int(len(hat)),
        "valid_rate": float(np.mean(best >= threshold)),
        "mean_best_iou": float(best.mean()),
        "class_counts": class_counts(hat, known),
    }
    if requested is not None:
        out["class_accuracy"] = accuracy(hat, requested)
    return out, renders, best


def _probe_split(x_train, y_train, x_test, y_test, classes, l2: float) -> dict:
    mu, sd = fit_standardizer(x_train)
    train = apply_standardizer(x_train, mu, sd)
    test = apply_standardizer(x_test, mu, sd)
    ridge = fit_ridge(train, y_train, classes, l2)
    centroids = fit_centroids(train, y_train, classes)
    return {
        "ridge": accuracy(predict_ridge(test, ridge, classes), y_test),
        "centroid": accuracy(predict_centroid(test, centroids, classes), y_test),
    }


def _denoise_summary(errors: np.ndarray) -> dict:
    return {"mean": float(errors.mean()), "median": float(np.median(errors))}


def run_mode(task: dict, mode: str, base_seed: int, out_dir: Path, log) -> dict:
    names = list(task["classes"])
    dim = int(task["dim"])
    hidden = int(task["hidden"])
    grid = int(task["grid"])
    n_train = int(task["train_per_class"])
    n_test = int(task["test_per_class"])
    threshold = float(task["iou_threshold"])
    coords = grid_coords(grid, dim)
    train_specs = sample_specs(
        names,
        n_train,
        np.random.default_rng(mix_seed(base_seed, task["name"], mode, "train-shapes")),
        dim,
        float(task["center_limit"]),
        float(task["scale_min"]),
        float(task["scale_max"]),
    )
    test_specs = sample_specs(
        names,
        n_test,
        np.random.default_rng(mix_seed(base_seed, task["name"], mode, "test-shapes")),
        dim,
        float(task["center_limit"]),
        float(task["scale_min"]),
        float(task["scale_max"]),
    )
    y_train = rasterize(train_specs, coords)
    y_test = rasterize(test_specs, coords)
    train_labels = _labels(names, n_train)
    test_labels = _labels(names, n_test)
    ens = init_ensemble(
        len(train_specs),
        dim,
        hidden,
        np.random.default_rng(mix_seed(base_seed, task["name"], mode, "init")),
        shared_init=(mode == "shared"),
    )
    log(f"  training {ens.n} {mode} INRs")
    train_ensemble(ens, coords, y_train, steps=int(task["inr_steps"]), lr=float(task["inr_lr"]))
    # Shared-init test nets start from the same vector as the training population.
    init_key = "init" if mode == "shared" else "test-init"
    test_ens = init_ensemble(
        len(test_specs),
        dim,
        hidden,
        np.random.default_rng(mix_seed(base_seed, task["name"], mode, init_key)),
        shared_init=(mode == "shared"),
    )
    train_ensemble(test_ens, coords, y_test, steps=int(task["inr_steps"]), lr=float(task["inr_lr"]))

    test_iou = _iou_to_target(test_ens, y_test, coords)
    rng_controls = np.random.default_rng(mix_seed(base_seed, task["name"], mode, "controls"))
    permuted = permute_neurons(test_ens, rng_controls)
    shuffled = shuffle_within_tensors(test_ens, np.random.default_rng(mix_seed(base_seed, task["name"], mode, "shuffle")))
    random_ens = random_like(test_ens, np.random.default_rng(mix_seed(base_seed, task["name"], mode, "random")))
    control_ious = {
        "trained": _iou_to_target(test_ens, y_test, coords),
        "permuted": _iou_to_target(permuted, y_test, coords),
        "shuffled": _iou_to_target(shuffled, y_test, coords),
        "random": _iou_to_target(random_ens, y_test, coords),
    }

    flat_train = flatten_weights(ens)
    flat_test = flatten_weights(test_ens)
    weight_mu, weight_sd = fit_standardizer(flat_train)
    x_train = apply_standardizer(flat_train, weight_mu, weight_sd)
    x_test = apply_standardizer(flat_test, weight_mu, weight_sd)
    x_perm = apply_standardizer(flatten_weights(permuted), weight_mu, weight_sd)
    x_shuf = apply_standardizer(flatten_weights(shuffled), weight_mu, weight_sd)
    x_rand = apply_standardizer(flatten_weights(random_ens), weight_mu, weight_sd)

    l2 = float(task["ridge_l2"])
    # Random and shuffled networks are not any of the shape classes. These probes can only
    # emit a class name, so "accuracy" against unrecognized is how often they abstain (they cannot).
    probes = {
        "raw": {
            "test": _probe_split(x_train, train_labels, x_test, test_labels, names, l2),
            "permuted": _probe_split(x_train, train_labels, x_perm, test_labels, names, l2),
            "shuffled": _probe_split(x_train, train_labels, x_shuf, test_labels, names, l2),
            "random_abstain": _probe_split(
                x_train, train_labels, x_rand, np.array(["unrecognized"] * len(test_labels)), names, l2
            ),
        }
    }
    canon_train = flatten_weights(canonicalize(ens))
    canon_test = flatten_weights(canonicalize(test_ens))
    canon_perm = flatten_weights(canonicalize(permuted))
    probes["canonical"] = {
        "test": _probe_split(canon_train, train_labels, canon_test, test_labels, names, l2),
        "permuted": _probe_split(canon_train, train_labels, canon_perm, test_labels, names, l2),
    }
    layer_scores = {}
    for layer_name, columns in layer_slices(dim, hidden).items():
        if layer_name.startswith("b"):
            continue
        layer_scores[layer_name] = _probe_split(
            x_train[:, columns], train_labels, x_test[:, columns], test_labels, names, l2
        )["ridge"]

    bank, bank_labels = template_bank(
        dim, grid, names, float(task["center_limit"]), float(task["scale_min"]), float(task["scale_max"])
    )
    diff_cfg = dict(task["diffusion"])
    diff_cfg["class_names"] = names
    log(f"  diffusion in PCA-{diff_cfg['pca_components']} ({mode})")
    pack = _fit_latent_diffusion(
        x_train,
        train_labels,
        diff_cfg,
        np.random.default_rng(mix_seed(base_seed, task["name"], mode, "diffusion")),
        int(diff_cfg["pca_components"]),
    )
    denoise_t = int(diff_cfg["denoise_timestep"])
    denoise = {}
    for tag, matrix, label_source in (
        ("real", x_test, test_labels),
        ("permuted", x_perm, test_labels),
        ("shuffled", x_shuf, test_labels),
        ("random", x_rand, test_labels),
    ):
        encoded = _encode(matrix, pack)
        index = np.array([names.index(str(label)) for label in label_source], dtype=np.int64)
        errors = denoising_se(
            pack["model"],
            encoded,
            index,
            pack["schedule"],
            denoise_t,
            int(diff_cfg["time_dim"]),
            len(names),
            np.random.default_rng(mix_seed(base_seed, task["name"], mode, "denoise", tag)),
        )
        denoise[tag] = _denoise_summary(errors)

    n_each = int(diff_cfg["samples_per_class"])
    requested_index = np.repeat(np.arange(len(names)), n_each)
    requested = [names[int(i)] for i in requested_index]
    drawn = sample(
        pack["model"],
        len(requested_index),
        requested_index,
        pack["schedule"],
        int(diff_cfg["time_dim"]),
        len(names),
        np.random.default_rng(mix_seed(base_seed, task["name"], mode, "sample")),
    )
    decoded = _decode(drawn, pack, weight_mu, weight_sd)
    sample_stats, sample_renders, _sample_iou = _score_renders(
        decoded, requested, dim, hidden, coords, bank, bank_labels, threshold
    )
    # Memorization: nearest training weight, and nearest training mask.
    weight_dist = np.linalg.norm(decoded[:, None, :] - flat_train[None, :, :], axis=-1).min(axis=1)
    train_pair = np.linalg.norm(flat_train[:, None, :] - flat_train[None, :, :], axis=-1)
    np.fill_diagonal(train_pair, np.inf)
    train_nn = float(np.median(train_pair.min(axis=1)))
    train_renders = (y_train >= 0.5).astype(np.float64)
    sample_f = sample_renders.astype(np.float64)
    inter = sample_f @ train_renders.T
    union = sample_f.sum(1, keepdims=True) + train_renders.sum(1)[None, :] - inter
    copy_iou = (inter / np.maximum(union, 1.0)).max(axis=1)

    gaussian = np.random.default_rng(mix_seed(base_seed, task["name"], mode, "gaussian")).normal(size=drawn.shape)
    gaussian_flat = _decode(gaussian, pack, weight_mu, weight_sd)
    gaussian_stats, _, _ = _score_renders(
        gaussian_flat, None, dim, hidden, coords, bank, bank_labels, threshold
    )

    extra = {}
    if task.get("full_dimensional_control") and mode == "shared":
        log("  full-dimensional DDPM control")
        extra["full_dimensional"] = _direct_diffusion(
            x_train, train_labels, names, diff_cfg, weight_mu, weight_sd, coords, bank, bank_labels, threshold, dim, hidden,
            np.random.default_rng(mix_seed(base_seed, task["name"], mode, "full-d")),
            steps=int(diff_cfg["full_dimensional_steps"]),
        )
    if task.get("shuffled_model_control") and mode == "shared":
        log("  DDPM trained on shuffled weights")
        shuf_train = shuffle_within_tensors(ens, np.random.default_rng(mix_seed(base_seed, task["name"], mode, "shuf-model")))
        shuf_flat = flatten_weights(shuf_train)
        shuf_mu, shuf_sd = fit_standardizer(shuf_flat)
        extra["trained_on_shuffled"] = _pca_sample_only(
            apply_standardizer(shuf_flat, shuf_mu, shuf_sd),
            train_labels,
            names,
            diff_cfg,
            shuf_mu,
            shuf_sd,
            coords,
            bank,
            bank_labels,
            threshold,
            dim,
            hidden,
            np.random.default_rng(mix_seed(base_seed, task["name"], mode, "shuf-diff")),
        )

    figures = {}
    fig_dir = out_dir / "figures"
    show = min(6, n_test)
    pick = np.array([(k % len(names)) * n_test + (k // len(names)) for k in range(show)])
    recon_path = fig_dir / f"recon_{task['name']}_{mode}.png"
    save_mask_grid(
        recon_path,
        [
            ("target", y_test[pick]),
            ("INR", (predict_proba(_subset(test_ens, pick), coords) >= 0.5).astype(np.float64)),
        ],
        grid,
        dim,
        f"{task['name']} {mode} init: target vs trained INR (test IoU {float(test_iou.mean()):.2f})",
    )
    figures["reconstruction"] = str(recon_path)
    if dim == 2 and mode == "shared":
        control_path = fig_dir / "controls_2d_shared.png"
        # First test net of each class.
        chosen = np.array([i * n_test for i in range(len(names))])
        save_mask_grid(
            control_path,
            [
                ("target", y_test[chosen]),
                ("trained", (predict_proba(_subset(test_ens, chosen), coords) >= 0.5).astype(float)),
                ("permuted", (predict_proba(_subset(permuted, chosen), coords) >= 0.5).astype(float)),
                ("shuffled", (predict_proba(_subset(shuffled, chosen), coords) >= 0.5).astype(float)),
                ("random", (predict_proba(_subset(random_ens, chosen), coords) >= 0.5).astype(float)),
            ],
            grid,
            dim,
            "Permutation keeps the shape. Shuffle and a random init do not.",
        )
        figures["controls"] = str(control_path)
        best_units = [int(np.argmax(test_iou[i * n_test : (i + 1) * n_test]) + i * n_test) for i in range(len(names))]
        hyper_path = fig_dir / "hyperplanes_2d_shared.png"
        save_hyperplanes(
            hyper_path,
            coords,
            [y_test[i] for i in best_units],
            [_one(test_ens, i) for i in best_units],
            [f"{names[i]} first-layer cuts" for i in range(len(names))],
            grid,
        )
        figures["hyperplanes"] = str(hyper_path)
    sample_path = fig_dir / f"samples_{task['name']}_{mode}.png"
    show_per = min(4, n_each)
    # Class-major sample order, so the grid is four requested circles, then squares, then triangles.
    picks = [class_index * n_each + offset for class_index in range(len(names)) for offset in range(show_per)]
    save_mask_grid(
        sample_path,
        [("sample", sample_renders[picks].astype(float))],
        grid,
        dim,
        f"{task['name']} {mode}: 4 per requested class, left to right. Valid {sample_stats['valid_rate']:.2f}, class acc {sample_stats.get('class_accuracy', float('nan')):.2f}",
    )
    figures["samples"] = str(sample_path)

    # PCA scatter of standardized weights, fit on train, project shuffled test.
    scatter_basis = x_train - x_train.mean(0)
    _, _, vt = np.linalg.svd(scatter_basis, full_matrices=False)
    def project(matrix: np.ndarray) -> np.ndarray:
        return (matrix - x_train.mean(0)) @ vt[:2].T

    return {
        "init": mode,
        "parameter_count": parameter_count(dim, hidden),
        "n_train": int(ens.n),
        "n_test": int(test_ens.n),
        "inr_test_iou": {"mean": float(test_iou.mean()), "per_class": _per_class(test_iou, names, n_test)},
        "controls_iou_to_own_target": {name: {"mean": float(values.mean())} for name, values in control_ious.items()},
        "weight_probes": probes,
        "ridge_by_layer": layer_scores,
        "diffusion_pca": {
            "components": int(pack["pca"]["k"]),
            "variance_explained": float(pack["pca"]["explained"]),
            "timesteps": int(diff_cfg["timesteps"]),
            "train_steps": int(diff_cfg["train_steps"]),
            "denoise_timestep": denoise_t,
            "denoise_mse": denoise,
            "samples": sample_stats,
            "latent_gaussian": gaussian_stats,
            "memorization": {
                "mean_nn_weight_distance": float(weight_dist.mean()),
                "median_train_loo_nn_distance": train_nn,
                "nn_distance_ratio": float(weight_dist.mean() / max(train_nn, 1e-12)),
                "mean_max_iou_to_train_mask": float(copy_iou.mean()),
            },
            **extra,
        },
        "figures": figures,
        "_plot": {
            "project_train": project(x_train),
            "train_labels": train_labels.tolist(),
            "project_shuffled": project(x_shuf),
            "classes": names,
        },
    }


def _direct_diffusion(x_train, train_labels, names, diff_cfg, weight_mu, weight_sd, coords, bank, bank_labels, threshold, dim, hidden, rng, steps: int) -> dict:
    label_index = np.array([names.index(str(label)) for label in train_labels], dtype=np.int64)
    schedule = linear_schedule(int(diff_cfg["timesteps"]), beta_end=float(diff_cfg["beta_end"]))
    model = train_denoiser(
        x_train,
        label_index,
        schedule,
        steps=steps,
        batch_size=int(min(diff_cfg["batch_size"], x_train.shape[0])),
        hidden=int(diff_cfg["hidden"]),
        lr=float(diff_cfg["lr"]),
        time_dim=int(diff_cfg["time_dim"]),
        n_classes=len(names),
        rng=rng,
    )
    n_each = int(diff_cfg["samples_per_class"])
    requested_index = np.repeat(np.arange(len(names)), n_each)
    requested = [names[int(i)] for i in requested_index]
    drawn = sample(model, len(requested_index), requested_index, schedule, int(diff_cfg["time_dim"]), len(names), rng)
    decoded = drawn * weight_sd + weight_mu
    stats, _, _ = _score_renders(decoded, requested, dim, hidden, coords, bank, bank_labels, threshold)
    stats["train_steps"] = steps
    stats["note"] = "Epsilon model on the full standardized weight vector, no PCA."
    return stats


def _pca_sample_only(x_std, train_labels, names, diff_cfg, weight_mu, weight_sd, coords, bank, bank_labels, threshold, dim, hidden, rng) -> dict:
    pack = _fit_latent_diffusion(x_std, train_labels, {**diff_cfg, "class_names": names}, rng, int(diff_cfg["pca_components"]))
    n_each = int(diff_cfg["samples_per_class"])
    requested_index = np.repeat(np.arange(len(names)), n_each)
    requested = [names[int(i)] for i in requested_index]
    drawn = sample(
        pack["model"], len(requested_index), requested_index, pack["schedule"], int(diff_cfg["time_dim"]), len(names), rng
    )
    decoded = _decode(drawn, pack, weight_mu, weight_sd)
    stats, _, _ = _score_renders(decoded, requested, dim, hidden, coords, bank, bank_labels, threshold)
    stats["variance_explained"] = float(pack["pca"]["explained"])
    stats["note"] = "Same DDPM, but every training network had its tensors shuffled first."
    return stats


def run(config: dict, out_dir: Path, log=print) -> dict:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "version": __version__,
        "seed": int(config["seed"]),
        "iou_threshold": float(config["iou_threshold"]),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "tasks": {},
    }
    plots = []
    for task in config["tasks"]:
        task = dict(task)
        task["iou_threshold"] = float(config["iou_threshold"])
        task["ridge_l2"] = float(config["ridge_l2"])
        log(f"task {task['name']}")
        modes = {}
        for mode in task["init_modes"]:
            result = run_mode(task, mode, int(config["seed"]), out_dir, log)
            plots.append((task["name"], mode, result.pop("_plot")))
            modes[mode] = result
        summary["tasks"][task["name"]] = modes
    # Combined PCA figure for 2d, if both modes exist.
    twod = [item for item in plots if item[0] == "2d"]
    if twod:
        save_pca(
            out_dir / "figures" / "pca_2d.png",
            [
                {
                    "xy": item[2]["project_train"],
                    "labels": item[2]["train_labels"],
                    "classes": item[2]["classes"],
                    "extra_xy": item[2]["project_shuffled"],
                    "extra_label": "shuffled test",
                    "title": f"2d {item[1]} init",
                }
                for item in twod
            ],
        )
        shared = summary["tasks"]["2d"].get("shared")
        indep = summary["tasks"]["2d"].get("independent")
        if shared and indep:
            save_bars(
                out_dir / "figures" / "metrics_2d.png",
                "2d weight-space controls",
                ["trained", "permuted", "shuffled", "random"],
                {
                    "shared IoU": [shared["controls_iou_to_own_target"][k]["mean"] for k in ("trained", "permuted", "shuffled", "random")],
                    "independent IoU": [indep["controls_iou_to_own_target"][k]["mean"] for k in ("trained", "permuted", "shuffled", "random")],
                    "shared ridge": [
                        shared["weight_probes"]["raw"]["test"]["ridge"],
                        shared["weight_probes"]["raw"]["permuted"]["ridge"],
                        shared["weight_probes"]["raw"]["shuffled"]["ridge"],
                        shared["weight_probes"]["raw"]["random_abstain"]["ridge"],
                    ],
                },
                "score",
            )
    summary["runtime_sec"] = round(time.perf_counter() - started, 3)
    payload = to_jsonable(summary)
    (out_dir / "local_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    log(f"wrote {out_dir / 'local_summary.json'} in {summary['runtime_sec']}s")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train shape INRs and a DDPM over their weights.")
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    run(config, Path(args.out))


if __name__ == "__main__":
    main()
