"""Fair prompt for the shape-reading probe. The condition name never appears."""

from __future__ import annotations

import json
import math

import numpy as np

from wsp.mlp import parameter_count, unflatten_weights

LABELS = ("circle", "square", "triangle", "unrecognized")

READOUT_SCHEMA = {
    "name": "shape_readout",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {
                "type": "string",
                "enum": list(LABELS),
                "description": "Filled shape encoded by these weights, or unrecognized.",
            },
            "confidence": {
                "type": "number",
                "description": "Your probability, from 0 to 1, that label is correct.",
            },
            "evidence": {
                "type": "string",
                "description": "One short sentence about the weights. No step-by-step reasoning.",
            },
        },
        "required": ["label", "confidence", "evidence"],
    },
}


def estimate_tokens(text: str) -> int:
    """Conservative count for a cost ceiling. Decimal dumps are about two characters per token."""

    return max(1, int(math.ceil(len(text) / 2.0)))


def format_weights(flat: np.ndarray, in_dim: int, hidden: int, decimals: int = 4) -> str:
    ens = unflatten_weights(flat.reshape(1, -1), in_dim, hidden)
    specs = (
        ("layer1.weight", ens.w1[0]),
        ("layer1.bias", ens.b1[0]),
        ("layer2.weight", ens.w2[0]),
        ("layer2.bias", ens.b2[0]),
        ("layer3.weight", ens.w3[0]),
        ("layer3.bias", ens.b3[0]),
    )
    lines = [f"parameter_count: {parameter_count(in_dim, hidden)}"]
    for name, array in specs:
        rounded = np.round(array, decimals)
        rounded[rounded == 0] = 0.0
        if rounded.ndim == 1:
            body = ", ".join(f"{value:.4f}" for value in rounded)
            lines.append(f"{name}: {body}")
        else:
            lines.append(f"{name}:")
            for row in rounded:
                lines.append("  " + ", ".join(f"{value:.4f}" for value in row))
    return "\n".join(lines)


def messages_for(flat: np.ndarray, in_dim: int, hidden: int, decimals: int = 4) -> list[dict]:
    weights = format_weights(flat, in_dim, hidden, decimals)
    system = (
        "You read the weights of a multilayer perceptron. You have no tools and you must not "
        "simulate a forward pass or invent inputs to evaluate. Use only the numbers below."
    )
    user = f"""The network maps a point (x, y) in [-1, 1] to one logit. After a sigmoid, the logit was trained to be high inside one filled shape and low outside it. One shape per network.

Architecture, in order:
- Linear {in_dim} -> {hidden}, then ReLU
- Linear {hidden} -> {hidden}, then ReLU
- Linear {hidden} -> 1, the logit, with no activation

Matrices are row-major. Each layer is its weight matrix, then its bias. Values are rounded to {decimals} decimal places. That rounding is the full precision you get.

{weights}

Which filled shape do these weights encode? If they do not look like a network trained on one of the shapes, answer unrecognized.
Labels you may use: circle, square, triangle, unrecognized.
confidence is your probability that the label is correct, from 0 to 1.
evidence is one short sentence about the weights, not a calculation trace.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def prompt_tokens(flat: np.ndarray, in_dim: int, hidden: int, decimals: int = 4) -> int:
    return sum(estimate_tokens(message["content"]) for message in messages_for(flat, in_dim, hidden, decimals))


def parse_readout(content: str) -> dict | None:
    """Return label and confidence, or None. Evidence is intentionally dropped."""

    text = (content or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    label = str(obj.get("label", "")).strip().lower()
    if label not in LABELS:
        return None
    try:
        confidence = float(obj["confidence"])
    except (KeyError, TypeError, ValueError):
        return None
    if confidence < 0.0 or confidence > 1.0 or math.isnan(confidence):
        return None
    return {"label": label, "confidence": confidence}
