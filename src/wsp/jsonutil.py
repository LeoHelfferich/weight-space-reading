"""Small helpers for stable JSON summaries."""

from __future__ import annotations

import math

import numpy as np


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.floating):
        return to_jsonable(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    raise TypeError(f"cannot encode {type(obj)}")
