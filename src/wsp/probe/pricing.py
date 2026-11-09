"""OpenRouter price ceilings. Premium speed tiers are excluded before any call."""

from __future__ import annotations

from dataclasses import dataclass

PREMIUM_TIERS = {"priority", "ultrafast", "fast"}
HEADLINE_SLACK = 1.25


@dataclass(frozen=True)
class ModelPrice:
    model_id: str
    ceiling_prompt: float
    ceiling_completion: float
    cheap_prompt: float
    cheap_completion: float
    max_price_per_million: dict
    temperature_supported: bool
    reasoning_efforts: tuple[str, ...]
    source: str

    def worst_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.ceiling_prompt + completion_tokens * self.ceiling_completion

    def expected_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.cheap_prompt + completion_tokens * self.cheap_completion


def is_premium_endpoint(endpoint: dict) -> bool:
    tier = str(endpoint.get("service_tier") or "").lower()
    tag = str(endpoint.get("tag") or "").lower()
    tail = tag.split("/")[-1]
    return tier in PREMIUM_TIERS or tail in PREMIUM_TIERS


def _unit_prices(pricing: dict, prompt_tokens: int) -> tuple[float, float] | None:
    try:
        prompt = float(pricing["prompt"])
        completion = float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    if prompt <= 0 or completion <= 0:
        return None
    for override in pricing.get("overrides") or []:
        try:
            minimum = int(override.get("min_prompt_tokens", 10**18))
        except (TypeError, ValueError):
            continue
        if prompt_tokens >= minimum:
            prompt = max(prompt, float(override["prompt"]))
            completion = max(completion, float(override["completion"]))
    return prompt, completion


def allowed_endpoints(
    endpoints: list[dict],
    headline_prompt: float | None,
    headline_completion: float | None,
    prompt_tokens: int,
) -> list[tuple[dict, float, float]]:
    chosen = []
    for endpoint in endpoints:
        if is_premium_endpoint(endpoint):
            continue
        prices = _unit_prices(endpoint.get("pricing") or {}, prompt_tokens)
        if prices is None:
            continue
        prompt, completion = prices
        if headline_prompt and prompt > headline_prompt * HEADLINE_SLACK:
            continue
        if headline_completion and completion > headline_completion * HEADLINE_SLACK:
            continue
        chosen.append((endpoint, prompt, completion))
    return chosen


def price_from_endpoints(
    model_id: str,
    endpoints: list[dict],
    headline_prompt: float | None,
    headline_completion: float | None,
    reasoning_efforts: list[str],
    catalogue_parameters: list[str],
    prompt_tokens_for_override: int = 4096,
) -> ModelPrice | None:
    chosen = allowed_endpoints(endpoints, headline_prompt, headline_completion, prompt_tokens_for_override)
    if not chosen:
        return None
    ceiling_prompt = max(prompt for _, prompt, _ in chosen)
    ceiling_completion = max(completion for _, _, completion in chosen)
    cheap_prompt, cheap_completion = min(
        ((prompt, completion) for _, prompt, completion in chosen),
        key=lambda pair: pair[0] + pair[1],
    )
    parameters = set(catalogue_parameters)
    for endpoint, _, _ in chosen:
        parameters.update(endpoint.get("supported_parameters") or [])
    return ModelPrice(
        model_id=model_id,
        ceiling_prompt=ceiling_prompt,
        ceiling_completion=ceiling_completion,
        cheap_prompt=cheap_prompt,
        cheap_completion=cheap_completion,
        max_price_per_million={
            "prompt": round(ceiling_prompt * 1_000_000, 4) + 0.01,
            "completion": round(ceiling_completion * 1_000_000, 4) + 0.01,
        },
        temperature_supported="temperature" in parameters,
        reasoning_efforts=tuple(reasoning_efforts),
        source="openrouter",
    )


def price_from_pin(model_id: str, pin: dict) -> ModelPrice:
    return ModelPrice(
        model_id=model_id,
        ceiling_prompt=float(pin["ceiling_prompt_per_token"]),
        ceiling_completion=float(pin["ceiling_completion_per_token"]),
        cheap_prompt=float(pin["cheap_prompt_per_token"]),
        cheap_completion=float(pin["cheap_completion_per_token"]),
        max_price_per_million={
            "prompt": float(pin["max_price_per_million"]["prompt"]),
            "completion": float(pin["max_price_per_million"]["completion"]),
        },
        temperature_supported=bool(pin.get("temperature_supported", False)),
        reasoning_efforts=tuple(pin.get("reasoning_efforts", ("low", "medium", "high"))),
        source="pinned",
    )


def usd_to_eur(amount_usd: float, rate: float) -> float:
    return amount_usd * rate
