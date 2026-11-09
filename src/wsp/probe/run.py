"""One command for the paid read-the-weights probe.

    python -m wsp.probe --config configs/probe.json --dry-run
    python -m wsp.probe --config configs/probe.json --mock
    python -m wsp.probe --config configs/probe.json --cap-eur 3

Live mode reads OPENROUTER_API_KEY from the environment and never prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from wsp.controls import canonicalize, permute_neurons, rounded_flat, shuffle_within_tensors
from wsp.jsonutil import to_jsonable
from wsp.metrics import (
    accuracy,
    batched_iou,
    best_template,
    brier_score,
    confusion,
    expected_calibration_error,
    fit_centroids,
    fit_standardizer,
    predict_centroid,
    reject_threshold,
)
from wsp.mlp import init_ensemble, predict_proba, train_ensemble, unflatten_weights
from wsp.probe.client import HARD_CAP_EUR, ProbeClient, urllib_transport
from wsp.probe.pricing import price_from_endpoints, price_from_pin, usd_to_eur
from wsp.probe.prompt import LABELS, messages_for, parse_readout, prompt_tokens
from wsp.shapes import grid_coords, rasterize, sample_specs, template_bank


def _one_index(ens, index: int):
    return type(ens)(*(array[index : index + 1] for array in ens.arrays()))


def _train_bank(config: dict, mode: str):
    names = list(config["classes"])
    n_per = int(config["bank_per_class"])
    hidden = int(config["hidden"])
    grid = int(config["grid"])
    coords = grid_coords(grid, 2)
    specs = sample_specs(
        names,
        n_per,
        np.random.default_rng(_seed(config, mode, "shapes")),
        2,
        float(config["center_limit"]),
        float(config["scale_min"]),
        float(config["scale_max"]),
    )
    target = rasterize(specs, coords)
    ens = init_ensemble(
        len(specs),
        2,
        hidden,
        np.random.default_rng(_seed(config, mode, "init")),
        shared_init=(mode == "shared"),
    )
    train_ensemble(ens, coords, target, steps=int(config["inr_steps"]), lr=float(config["inr_lr"]))
    labels = np.repeat(np.array(names, dtype=object), n_per)
    return ens, labels, target, coords


def _seed(config: dict, *parts: object) -> int:
    from wsp.local_experiment import mix_seed

    return mix_seed(int(config["seed"]), "probe", *parts)


def _flat_of(ens, decimals: int) -> np.ndarray:
    return rounded_flat(ens, decimals)[0]


def build_items(config: dict) -> tuple[list[dict], dict]:
    names = list(config["classes"])
    n_per = int(config["bank_per_class"])
    decimals = int(config["decimals"])
    hidden = int(config["hidden"])
    per = config["per_class"]
    shared, _shared_labels, shared_target, coords = _train_bank(config, "shared")
    indep, _indep_labels, indep_target, _indep_coords = _train_bank(config, "independent")
    items: list[dict] = []

    def add(flat: np.ndarray, condition: str, truth: str, shape: str, replicate: int, source_mask) -> None:
        items.append(
            {
                "id": f"{condition}-{shape}-{replicate}",
                "condition": condition,
                "truth": truth,
                "shape": shape,
                "replicate": replicate,
                "flat": np.asarray(flat, dtype=np.float64),
                "source_mask": None if source_mask is None else np.asarray(source_mask, dtype=bool),
            }
        )

    for name in names:
        base = names.index(name) * n_per
        for replicate in range(int(per["real_shared"])):
            mask = shared_target[base + replicate] >= 0.5
            add(_flat_of(_one_index(shared, base + replicate), decimals), "real_shared", name, name, replicate, mask)
        for replicate in range(int(per["real_independent"])):
            mask = indep_target[base + replicate] >= 0.5
            add(_flat_of(_one_index(indep, base + replicate), decimals), "real_independent", name, name, replicate, mask)
        for replicate in range(int(per["permuted"])):
            donor = _one_index(shared, base + replicate)
            flipped = permute_neurons(donor, np.random.default_rng(_seed(config, "permute", name, replicate)))
            add(_flat_of(flipped, decimals), "permuted", name, name, replicate, shared_target[base + replicate] >= 0.5)
        for replicate in range(int(per["shuffled"])):
            donor = _one_index(shared, base + replicate)
            mixed = shuffle_within_tensors(donor, np.random.default_rng(_seed(config, "shuffle", name, replicate)))
            add(_flat_of(mixed, decimals), "shuffled", "unrecognized", name, replicate, shared_target[base + replicate] >= 0.5)
        for replicate in range(int(per["random"])):
            fresh = init_ensemble(1, 2, hidden, np.random.default_rng(_seed(config, "random", name, replicate)))
            add(_flat_of(fresh, decimals), "random", "unrecognized", name, replicate, None)

    order = np.random.default_rng(_seed(config, "order")).permutation(len(items))
    items = [items[int(i)] for i in order]
    held = (
        set(range(int(per["real_shared"])))
        | set(range(int(per["real_independent"])))
        | set(range(int(per["permuted"])))
        | set(range(int(per["shuffled"])))
    )
    baseline_rows = []
    baseline_labels = []
    for class_index, name in enumerate(names):
        for replicate in range(n_per):
            if replicate in held:
                continue
            baseline_rows.append(_flat_of(_one_index(shared, class_index * n_per + replicate), decimals))
            baseline_labels.append(name)
            baseline_rows.append(_flat_of(_one_index(indep, class_index * n_per + replicate), decimals))
            baseline_labels.append(name)
    bank_masks, bank_labels = template_bank(
        2,
        int(config["grid"]),
        names,
        float(config["center_limit"]),
        float(config["scale_min"]),
        float(config["scale_max"]),
    )
    context = {
        "baseline_x": np.stack(baseline_rows, axis=0),
        "baseline_y": np.array(baseline_labels, dtype=object),
        "coords": coords,
        "bank_masks": bank_masks,
        "bank_labels": bank_labels,
    }
    pred = predict_proba(shared, coords) >= 0.5
    inter = np.logical_and(pred, shared_target >= 0.5).sum(1)
    union = np.logical_or(pred, shared_target >= 0.5).sum(1)
    context["shared_train_iou"] = float(np.mean(inter / np.maximum(union, 1)))
    return items, context


def canonical_matrix(flats: np.ndarray, hidden: int, decimals: int) -> np.ndarray:
    rows = []
    for flat in flats:
        ens = unflatten_weights(flat.reshape(1, -1), 2, hidden)
        rows.append(rounded_flat(canonicalize(ens), decimals)[0])
    return np.stack(rows, axis=0)


def score_baselines(items: list[dict], context: dict, config: dict) -> dict:
    hidden = int(config["hidden"])
    decimals = int(config["decimals"])
    classes = list(config["classes"])
    x_train = context["baseline_x"]
    y_train = context["baseline_y"]
    mu, sd = fit_standardizer(x_train)
    train = (x_train - mu) / sd
    centroids = fit_centroids(train, y_train, classes)
    threshold = reject_threshold(train, y_train, centroids, classes)
    raw = []
    canon_train = canonical_matrix(x_train, hidden, decimals)
    c_mu, c_sd = fit_standardizer(canon_train)
    c_train = (canon_train - c_mu) / c_sd
    c_centroids = fit_centroids(c_train, y_train, classes)
    c_threshold = reject_threshold(c_train, y_train, c_centroids, classes)
    canon_pred = []
    forward = []
    truths = []
    for item in items:
        flat = item["flat"]
        scaled = (flat - mu) / sd
        raw.append(predict_centroid(scaled.reshape(1, -1), centroids, classes, threshold)[0])
        canon_flat = canonical_matrix(flat.reshape(1, -1), hidden, decimals)[0]
        canon_pred.append(
            predict_centroid(((canon_flat - c_mu) / c_sd).reshape(1, -1), c_centroids, classes, c_threshold)[0]
        )
        ens = unflatten_weights(flat.reshape(1, -1), 2, hidden)
        render = predict_proba(ens, context["coords"]) >= 0.5
        if item["source_mask"] is not None:
            # Own-target IoU. The coarse template bank confuses circles with squares, so it is not the ceiling.
            source_iou = float(batched_iou(render, item["source_mask"][None, :])[0])
            guess = item["shape"] if source_iou >= float(config["iou_threshold"]) else "unrecognized"
        else:
            hat, _best = best_template(render, context["bank_masks"], context["bank_labels"], float(config["iou_threshold"]))
            guess = hat[0]
        forward.append(guess)
        truths.append(item["truth"])
    always_unrecognized = ["unrecognized"] * len(items)
    return {
        "n_baseline_networks": int(len(y_train)),
        "reject_distance_raw": threshold,
        "nearest_centroid_raw": accuracy(raw, truths),
        "nearest_centroid_canonical": accuracy(canon_pred, truths),
        "always_unrecognized": accuracy(always_unrecognized, truths),
        "uniform_chance": 1.0 / len(LABELS),
        "forward_pass_ceiling": accuracy(forward, truths),
        "by_condition": _by_condition(items, {"centroid_raw": raw, "centroid_canonical": canon_pred, "forward": forward}),
    }


def _by_condition(items: list[dict], columns: dict[str, list[str]]) -> dict:
    out = {}
    for condition in dict.fromkeys(item["condition"] for item in items):
        index = [i for i, item in enumerate(items) if item["condition"] == condition]
        truths = [items[i]["truth"] for i in index]
        out[condition] = {name: accuracy([values[i] for i in index], truths) for name, values in columns.items()}
    return out


def mock_label(item: dict) -> tuple[str, float]:
    """Deterministic stand-in. Not a model and not a result."""

    if item["condition"] in {"shuffled", "random"}:
        return "unrecognized", 0.8
    if item["condition"] == "real_shared" and item["shape"] == "square" and int(item["replicate"]) == 0:
        return "triangle", 0.8
    return str(item["truth"]), 0.8


class MockTransport:
    def __init__(self, items_by_id: dict[str, dict]):
        self.items_by_id = items_by_id
        self.current_id: str | None = None
        self.calls = 0

    def __call__(self, payload: dict):
        self.calls += 1
        if "Authorization" in payload or "api_key" in payload:
            raise RuntimeError("mock payload must not carry credentials")
        label, confidence = mock_label(self.items_by_id[self.current_id])
        content = json.dumps({"label": label, "confidence": confidence, "evidence": "mock"})
        usage = {"prompt_tokens": 12, "completion_tokens": 24, "cost": 0.0}
        return 200, {"choices": [{"message": {"content": content}}], "usage": usage}


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "wsp-experiment"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def resolve_fx(config: dict, fetch, env: dict) -> tuple[float, str, str | None]:
    override = env.get("WSP_USD_TO_EUR")
    if override:
        return float(override), "env", None
    try:
        data = fetch("https://api.frankfurter.app/latest?from=USD&to=EUR")
        return float(data["rates"]["EUR"]), "frankfurter", data.get("date")
    except Exception:
        return float(config["pinned_usd_to_eur"]), "pinned", config.get("pinned_fx_date")


def resolve_prices(config: dict, model_ids: list[str], fetch, require_live: bool) -> tuple[dict, str]:
    pins = config["pinned_pricing"]
    try:
        catalogue = fetch("https://openrouter.ai/api/v1/models")
        by_id = {entry["id"]: entry for entry in catalogue.get("data", [])}
    except Exception as exc:
        if require_live:
            raise SystemExit(
                "Refusing paid calls because the OpenRouter model list could not be read "
                f"({exc.__class__.__name__}). No request was sent."
            )
        return {model_id: price_from_pin(model_id, pins[model_id]) for model_id in model_ids}, "pinned"
    prices = {}
    for model_id in model_ids:
        entry = by_id.get(model_id)
        if entry is None:
            if require_live:
                raise SystemExit(f"Refusing paid calls: {model_id} is not in the OpenRouter catalogue.")
            prices[model_id] = price_from_pin(model_id, pins[model_id])
            continue
        try:
            endpoints_payload = fetch(f"https://openrouter.ai/api/v1/models/{model_id}/endpoints")
            endpoints = endpoints_payload["data"]["endpoints"]
        except Exception as exc:
            if require_live:
                raise SystemExit(
                    f"Refusing paid calls: endpoint prices for {model_id} could not be read ({exc.__class__.__name__})."
                )
            prices[model_id] = price_from_pin(model_id, pins[model_id])
            continue
        headline = entry.get("pricing") or {}
        try:
            headline_prompt = float(headline["prompt"])
            headline_completion = float(headline["completion"])
        except (KeyError, TypeError, ValueError):
            headline_prompt = headline_completion = None
        efforts = list((entry.get("reasoning") or {}).get("supported_efforts") or pins[model_id]["reasoning_efforts"])
        resolved = price_from_endpoints(
            model_id,
            endpoints,
            headline_prompt,
            headline_completion,
            efforts,
            list(entry.get("supported_parameters") or []),
        )
        if resolved is None:
            if require_live:
                raise SystemExit(f"Refusing paid calls: no non-premium endpoint for {model_id}.")
            prices[model_id] = price_from_pin(model_id, pins[model_id])
        else:
            prices[model_id] = resolved
    source = "openrouter" if all(price.source == "openrouter" for price in prices.values()) else "pinned"
    return prices, source


def _plan(items, config, prices, rate: float) -> dict:
    hidden = int(config["hidden"])
    decimals = int(config["decimals"])
    by_model = {}
    worst = 0.0
    expected = 0.0
    for model_id, spec in config["models"].items():
        worst_model = 0.0
        expected_model = 0.0
        tokens = []
        for item in items:
            count = prompt_tokens(item["flat"], 2, hidden, decimals)
            tokens.append(count)
            price = prices[model_id]
            worst_model += usd_to_eur(price.worst_usd(count, int(spec["max_tokens"])), rate)
            expected_model += usd_to_eur(price.expected_usd(count, int(spec["expected_completion_tokens"])), rate)
        by_model[model_id] = {
            "calls": len(items),
            "max_tokens": int(spec["max_tokens"]),
            "expected_completion_tokens": int(spec["expected_completion_tokens"]),
            "reasoning_effort": spec["reasoning_effort"],
            "mean_prompt_tokens_estimate": float(np.mean(tokens)),
            "worst_case_eur": worst_model,
            "expected_eur": expected_model,
            "temperature_supported": prices[model_id].temperature_supported,
            "price_source": prices[model_id].source,
            "ceiling_prompt_per_million": prices[model_id].ceiling_prompt * 1_000_000,
            "ceiling_completion_per_million": prices[model_id].ceiling_completion * 1_000_000,
            "cheap_prompt_per_million": prices[model_id].cheap_prompt * 1_000_000,
            "cheap_completion_per_million": prices[model_id].cheap_completion * 1_000_000,
        }
        worst += worst_model
        expected += expected_model
    return {"worst_case_eur": worst, "expected_eur": expected, "by_model": by_model}


def _metrics_for(records: list[dict]) -> dict:
    if not records:
        return {}
    correct = np.array([1.0 if record["correct"] else 0.0 for record in records])
    parsed = [record for record in records if record["parse_ok"]]
    conf = np.array([record["confidence"] for record in parsed], dtype=np.float64)
    parsed_correct = np.array([1.0 if record["correct"] else 0.0 for record in parsed], dtype=np.float64)
    labels = list(LABELS)
    by_condition = {}
    for condition in dict.fromkeys(record["condition"] for record in records):
        subset = [record for record in records if record["condition"] == condition]
        by_condition[condition] = {
            "n": len(subset),
            "accuracy": float(np.mean([record["correct"] for record in subset])),
        }
    return {
        "n": len(records),
        "n_parsed": len(parsed),
        "accuracy": float(correct.mean()),
        "brier": None if len(parsed) == 0 else brier_score(conf, parsed_correct),
        "calibration": None if len(parsed) == 0 else expected_calibration_error(conf, parsed_correct),
        "confusion": confusion(
            [record["prediction"] or "unrecognized" for record in records],
            [record["truth"] for record in records],
            labels,
        ),
        "labels": labels,
        "by_condition": by_condition,
    }


def format_report(summary: dict) -> str:
    lines = [
        f"mode: {summary['mode']}",
        f"pricing: {summary['pricing_source']}    fx: {summary['fx_source']} {summary.get('fx_date') or ''} USD→EUR {summary['usd_to_eur']:.5f}",
        f"items: {summary['n_items']}    calls planned: {summary['n_calls_planned']}",
        f"worst-case reserve: {summary['plan']['worst_case_eur']:.4f} EUR",
        f"expected if the cheap tier and short answers: {summary['plan']['expected_eur']:.4f} EUR",
        f"cap: {summary['cap_eur']:.2f} EUR (hard max {HARD_CAP_EUR:.0f})",
    ]
    for model_id, row in summary["plan"]["by_model"].items():
        lines.append(
            f"  {model_id}: prompt≈{row['mean_prompt_tokens_estimate']:.0f} tok, "
            f"max_tokens={row['max_tokens']}, effort={row['reasoning_effort']}, "
            f"worst {row['worst_case_eur']:.4f} EUR, expected {row['expected_eur']:.4f} EUR, "
            f"temperature_supported={row['temperature_supported']}"
        )
    if summary["mode"] != "dry-run":
        lines.append(
            f"spent: {summary['spent_eur']:.4f} EUR    "
            f"succeeded {summary['calls']['succeeded']}  cached {summary['calls']['cached']}  "
            f"failed {summary['calls']['failed']}  not_sent {summary['calls']['not_sent']}"
        )
        lines.append(f"stop: {summary['stopped_reason']}")
    baselines = summary.get("baselines")
    if baselines:
        lines.append(
            "local baselines on these same items: "
            f"forward pass {baselines['forward_pass_ceiling']:.2f}, "
            f"centroid {baselines['nearest_centroid_raw']:.2f}, "
            f"centroid after neuron sort {baselines['nearest_centroid_canonical']:.2f}, "
            f"always unrecognized {baselines['always_unrecognized']:.2f}, "
            f"chance {baselines['uniform_chance']:.2f}"
        )
    lines.append(
        "Claude Opus 5.5 bills a minimum reasoning budget of 1024 tokens, so its expected "
        "completion length stays near that floor. The cap reserves max_tokens at the dearest non-premium price."
    )
    return "\n".join(lines)


def run_probe(
    config: dict,
    mode: str,
    cap_eur: float,
    out_dir: Path | None = None,
    fetch=fetch_json,
    env: dict | None = None,
    sleeper=time.sleep,
    transport=None,
) -> dict:
    if mode not in {"dry-run", "mock", "live"}:
        raise ValueError(f"unknown mode {mode}")
    if cap_eur <= 0 or cap_eur > HARD_CAP_EUR:
        raise ValueError(f"--cap-eur must be in (0, {HARD_CAP_EUR:g}]")
    env = dict(os.environ if env is None else env)
    model_ids = list(config["models"])
    for model_id, spec in config["models"].items():
        if spec["reasoning_effort"] not in config["pinned_pricing"][model_id]["reasoning_efforts"]:
            raise ValueError(f"{model_id} effort {spec['reasoning_effort']} is not in the pinned effort list")
    items, context = build_items(config)
    baselines = score_baselines(items, context, config)
    rate, fx_source, fx_date = resolve_fx(config, fetch, env)
    prices, pricing_source = resolve_prices(config, model_ids, fetch, require_live=(mode == "live"))
    for model_id, spec in config["models"].items():
        efforts = prices[model_id].reasoning_efforts
        if efforts and spec["reasoning_effort"] not in efforts:
            raise SystemExit(
                f"Refusing to call {model_id}: effort {spec['reasoning_effort']!r} is not in {efforts}."
            )
    plan = _plan(items, config, prices, rate)
    summary = {
        "mode": mode,
        "cap_eur": cap_eur,
        "hard_cap_eur": HARD_CAP_EUR,
        "usd_to_eur": rate,
        "fx_source": fx_source,
        "fx_date": fx_date,
        "pricing_source": pricing_source,
        "n_items": len(items),
        "n_calls_planned": len(items) * len(model_ids),
        "plan": plan,
        "baselines": baselines,
        "shared_train_iou": context["shared_train_iou"],
        "calls": {"succeeded": 0, "cached": 0, "failed": 0, "not_sent": 0},
        "spent_eur": 0.0,
        "stopped_reason": "dry-run" if mode == "dry-run" else "completed",
        "records": [],
    }
    if mode == "dry-run":
        return to_jsonable(summary)
    if plan["worst_case_eur"] > cap_eur + 1e-9 and mode == "live":
        # Live refuses to start when the reserve itself exceeds the cap, so no call is sent.
        summary["stopped_reason"] = "cap"
        summary["calls"]["not_sent"] = summary["n_calls_planned"]
        summary["note"] = "Worst-case reserve exceeds the cap, so no request was sent."
        _emit(summary, out_dir)
        return to_jsonable(summary)

    cache_dir = Path(env.get("WSP_CACHE_DIR") or config["cache_dir"])
    items_by_id = {item["id"]: item for item in items}
    if mode == "mock":
        mock = transport if isinstance(transport, MockTransport) else MockTransport(items_by_id)
        client = ProbeClient(mock, cache_dir, cap_eur, rate, sleeper=sleeper)
    else:
        key = env.get("OPENROUTER_API_KEY", "")
        if not key.strip():
            raise SystemExit("Set OPENROUTER_API_KEY, or run with --mock or --dry-run. No request was sent.")
        if transport is None:
            transport = urllib_transport(env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), key, 180)
        client = ProbeClient(transport, cache_dir, cap_eur, rate, sleeper=sleeper)
    hidden = int(config["hidden"])
    decimals = int(config["decimals"])
    temperature = config.get("temperature", None)
    try:
        for item in items:
            for model_id, spec in config["models"].items():
                if client.stop_reason:
                    summary["calls"]["not_sent"] += 1
                    continue
                if mode == "mock" and isinstance(client.transport, MockTransport):
                    client.transport.current_id = item["id"]
                messages = messages_for(item["flat"], 2, hidden, decimals)
                result = client.complete(
                    model_id,
                    messages,
                    int(spec["max_tokens"]),
                    spec["reasoning_effort"],
                    temperature,
                    prices[model_id],
                )
                if not result.ok:
                    if result.attempted:
                        summary["calls"]["failed"] += 1
                    else:
                        summary["calls"]["not_sent"] += 1
                    if result.stop_reason:
                        summary["stopped_reason"] = result.stop_reason
                        client.stop_reason = result.stop_reason
                    continue
                parsed = parse_readout(result.content)
                prediction = parsed["label"] if parsed else None
                confidence = parsed["confidence"] if parsed else None
                correct = bool(parsed) and prediction == item["truth"]
                summary["records"].append(
                    {
                        "id": item["id"],
                        "model": model_id,
                        "condition": item["condition"],
                        "truth": item["truth"],
                        "prediction": prediction,
                        "confidence": confidence,
                        "correct": correct,
                        "parse_ok": bool(parsed),
                        "cached": result.cached,
                        "cost_eur": result.cost_eur,
                    }
                )
                if result.cached:
                    summary["calls"]["cached"] += 1
                else:
                    summary["calls"]["succeeded"] += 1
                if result.stop_reason:
                    summary["stopped_reason"] = result.stop_reason
                    client.stop_reason = result.stop_reason
    finally:
        summary["spent_eur"] = client.spent_eur
        summary["metrics"] = {
            model_id: _metrics_for([record for record in summary["records"] if record["model"] == model_id])
            for model_id in model_ids
        }
        _emit(summary, out_dir)
    return to_jsonable(summary)


def _emit(summary: dict, out_dir: Path | None) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # Aggregates only. Raw completions stay in the gitignored cache.
    public = {key: value for key, value in summary.items() if key != "records"}
    public["records"] = [
        {key: value for key, value in record.items() if key != "evidence"} for record in summary.get("records", [])
    ]
    (out_dir / "summary.json").write_text(json.dumps(to_jsonable(public), indent=2) + "\n")


_PROBE_KEYS = (
    "seed",
    "hidden",
    "grid",
    "inr_steps",
    "inr_lr",
    "bank_per_class",
    "decimals",
    "iou_threshold",
    "classes",
    "center_limit",
    "scale_min",
    "scale_max",
    "per_class",
    "default_cap_eur",
    "cache_dir",
    "pinned_usd_to_eur",
    "models",
    "pinned_pricing",
)
_PER_CLASS_KEYS = ("real_shared", "real_independent", "permuted", "shuffled", "random")


def load_probe_config(path: str | Path) -> dict:
    """Load a probe JSON config the same way the command does, and check its keys."""

    config_path = Path(path)
    config = json.loads(config_path.read_text())
    missing = [key for key in _PROBE_KEYS if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing {missing}")
    per_class = config["per_class"]
    missing_conditions = [key for key in _PER_CLASS_KEYS if key not in per_class]
    if missing_conditions:
        raise ValueError(f"{config_path} per_class is missing {missing_conditions}")
    if not isinstance(config["models"], dict) or not config["models"]:
        raise ValueError(f"{config_path} has no models")
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask frontier models which shape an MLP encodes.")
    parser.add_argument("--config", default="configs/probe.json")
    parser.add_argument("--out", default="results/probe_runs")
    parser.add_argument("--cap-eur", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.mock:
        raise SystemExit("Choose one of --dry-run or --mock.")
    config = load_probe_config(args.config)
    cap = float(config["default_cap_eur"] if args.cap_eur is None else args.cap_eur)
    if cap <= 0 or cap > HARD_CAP_EUR:
        raise SystemExit(f"--cap-eur must be in (0, {HARD_CAP_EUR:g}].")
    mode = "dry-run" if args.dry_run else "mock" if args.mock else "live"
    try:
        summary = run_probe(config, mode, cap, Path(args.out) if mode != "dry-run" else None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(format_report(summary))
    if mode != "dry-run" and summary.get("metrics"):
        for model_id, metrics in summary["metrics"].items():
            if not metrics:
                continue
            brier = metrics["brier"]
            ece = None if not metrics["calibration"] else metrics["calibration"]["ece"]
            print(f"{model_id}: accuracy {metrics['accuracy']:.3f}  brier {brier}  ece {ece}")
    reason = summary["stopped_reason"]
    if reason in {"4xx", "auth", "network", "cost-exceeded-estimate"}:
        raise SystemExit(2)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
