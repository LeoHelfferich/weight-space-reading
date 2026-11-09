"""OpenRouter chat client with a hard euro cap, a response cache, and parameter fallbacks."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from wsp.probe.pricing import ModelPrice
from wsp.probe.prompt import READOUT_SCHEMA, estimate_tokens

_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+)", re.IGNORECASE)
HARD_CAP_EUR = 5.0


def redact(text: str) -> str:
    return _SECRET.sub("[REDACTED]", text)


def validate_base_url(url: str) -> str:
    parts = urllib.parse.urlparse(url)
    host = parts.hostname or ""
    if parts.scheme != "https" or host != "openrouter.ai":
        raise ValueError("OPENROUTER_BASE_URL must be https://openrouter.ai")
    return url.rstrip("/")


def cache_key(payload: dict) -> str:
    blob = json.dumps({"v": 1, "body": payload}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _error_text(body: dict) -> str:
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or body)


def is_temperature_rejection(status: int, body: dict) -> bool:
    return status == 400 and "temperature" in _error_text(body).lower()


def is_schema_rejection(status: int, body: dict) -> bool:
    text = _error_text(body).lower()
    return status == 400 and any(token in text for token in ("response_format", "json_schema", "structured output"))


@dataclass
class CallResult:
    ok: bool
    cached: bool
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_eur: float
    error: str | None = None
    stop_reason: str | None = None
    attempted: bool = False


class ProbeClient:
    def __init__(
        self,
        transport,
        cache_dir: Path,
        cap_eur: float,
        usd_to_eur_rate: float,
        sleeper=time.sleep,
    ):
        if cap_eur <= 0 or cap_eur > HARD_CAP_EUR:
            raise ValueError(f"cap-eur must be in (0, {HARD_CAP_EUR:g}]")
        self.transport = transport
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cap_eur = float(cap_eur)
        self.usd_to_eur_rate = float(usd_to_eur_rate)
        self.sleeper = sleeper
        self.spent_eur = 0.0
        self.consecutive_4xx = 0
        self.stop_reason: str | None = None
        self.drop_temperature: set[str] = set()
        self.drop_schema: set[str] = set()
        self._load_adaptations()

    def _adapt_path(self) -> Path:
        return self.cache_dir / "adaptations.json"

    def _load_adaptations(self) -> None:
        path = self._adapt_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.drop_temperature.update(data.get("drop_temperature") or [])
        self.drop_schema.update(data.get("drop_schema") or [])

    def _save_adaptations(self) -> None:
        payload = {
            "drop_temperature": sorted(self.drop_temperature),
            "drop_schema": sorted(self.drop_schema),
        }
        self._adapt_path().write_text(json.dumps(payload, indent=2) + "\n")

    def build_payload(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        effort: str,
        temperature: float | None,
        price: ModelPrice,
        send_temperature: bool,
        send_schema: bool,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "reasoning": {"effort": effort, "exclude": True},
            "provider": {
                "sort": "price",
                "allow_fallbacks": True,
                "max_price": price.max_price_per_million,
            },
        }
        if send_schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": READOUT_SCHEMA}
        if send_temperature and temperature is not None:
            payload["temperature"] = temperature
        return payload

    def estimate_eur(self, messages: list[dict], max_tokens: int, price: ModelPrice) -> float:
        tokens = sum(estimate_tokens(message["content"]) for message in messages)
        return price.worst_usd(tokens, max_tokens) * self.usd_to_eur_rate

    def complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        effort: str,
        temperature: float | None,
        price: ModelPrice,
    ) -> CallResult:
        if self.stop_reason:
            return CallResult(False, False, "", 0, 0, 0.0, error=self.stop_reason, stop_reason=self.stop_reason)
        send_temperature = (
            temperature is not None and price.temperature_supported and model not in self.drop_temperature
        )
        send_schema = model not in self.drop_schema
        payload = self.build_payload(
            model, messages, max_tokens, effort, temperature, price, send_temperature, send_schema
        )
        cached = self._read_cache(payload)
        if cached is not None:
            self.consecutive_4xx = 0
            return cached
        estimate = self.estimate_eur(messages, max_tokens, price)
        if self.spent_eur + estimate > self.cap_eur + 1e-9:
            self.stop_reason = "cap"
            return CallResult(
                False,
                False,
                "",
                0,
                0,
                0.0,
                error=f"next call estimated at {estimate:.4f} EUR would pass the {self.cap_eur:.2f} EUR cap",
                stop_reason="cap",
            )
        status, body, payload = self._send_with_fallbacks(
            model, messages, max_tokens, effort, temperature, price, payload, send_temperature, send_schema
        )
        if status is None:
            self.stop_reason = "network"
            return CallResult(
                False, False, "", 0, 0, 0.0, error="network", stop_reason="network", attempted=True
            )
        if status >= 400:
            self._note_4xx(status)
            return CallResult(
                False,
                False,
                "",
                0,
                0,
                0.0,
                error=redact(_error_text(body))[:300],
                stop_reason=self.stop_reason,
                attempted=True,
            )
        usage = body.get("usage") if isinstance(body, dict) else {}
        cost_usd, prompt_tokens, completion_tokens = _usage_cost(
            usage or {}, price.ceiling_prompt, price.ceiling_completion
        )
        cost_eur = cost_usd * self.usd_to_eur_rate
        if cost_eur > estimate * 1.5 + 0.02:
            self.stop_reason = "cost-exceeded-estimate"
        self.spent_eur += cost_eur
        self.consecutive_4xx = 0
        content = message_text(body)
        self._write_cache(payload, body, cost_usd, prompt_tokens, completion_tokens)
        return CallResult(
            True,
            False,
            content,
            prompt_tokens,
            completion_tokens,
            cost_eur,
            stop_reason=self.stop_reason,
            attempted=True,
        )

    def _send_with_fallbacks(
        self, model, messages, max_tokens, effort, temperature, price, payload, send_temperature, send_schema
    ):
        status, body = self._exchange(payload)
        if status == 400 and send_temperature and is_temperature_rejection(status, body):
            self.drop_temperature.add(model)
            self._save_adaptations()
            send_temperature = False
            payload = self.build_payload(
                model, messages, max_tokens, effort, temperature, price, False, send_schema
            )
            status, body = self._exchange(payload)
        if status == 400 and send_schema and is_schema_rejection(status, body):
            self.drop_schema.add(model)
            self._save_adaptations()
            payload = self.build_payload(
                model, messages, max_tokens, effort, temperature, price, send_temperature, False
            )
            status, body = self._exchange(payload)
        return status, body, payload

    def _exchange(self, payload: dict) -> tuple[int | None, dict]:
        delays = (0.0, 1.0, 2.0)
        last_error = "network"
        for attempt, delay in enumerate(delays):
            if delay:
                self.sleeper(delay)
            try:
                status, body = self.transport(payload)
            except Exception as exc:  # noqa: BLE001 — transport failures must not leak the key
                last_error = redact(str(exc))[:300]
                status, body = None, {"error": {"message": last_error}}
            if status is None or status >= 500:
                continue
            return status, body if isinstance(body, dict) else {"error": {"message": "non-object body"}}
        return None, {"error": {"message": last_error}}

    def _note_4xx(self, status: int) -> None:
        if status in (401, 403):
            self.stop_reason = "auth"
            return
        self.consecutive_4xx += 1
        if self.consecutive_4xx >= 3:
            self.stop_reason = "4xx"

    def _cache_path(self, payload: dict) -> Path:
        return self.cache_dir / f"{cache_key(payload)}.json"

    def _read_cache(self, payload: dict) -> CallResult | None:
        path = self._cache_path(payload)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not data.get("ok"):
            return None
        return CallResult(
            ok=True,
            cached=True,
            content=str(data.get("content") or ""),
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            cost_eur=0.0,
        )

    def _write_cache(self, payload, body, cost_usd, prompt_tokens, completion_tokens) -> None:
        path = self._cache_path(payload)
        record = {
            "ok": True,
            "content": message_text(body),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        }
        # Raw provider bodies stay in this gitignored directory. Do not commit them.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record) + "\n")
        temporary.replace(path)


def message_text(body: dict) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _usage_cost(usage: dict, ceiling_prompt: float, ceiling_completion: float) -> tuple[float, int, int]:
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    cost = usage.get("cost", None)
    if cost is not None:
        try:
            return float(cost), prompt_tokens, completion_tokens
        except (TypeError, ValueError):
            pass
    return (
        prompt_tokens * ceiling_prompt + completion_tokens * ceiling_completion,
        prompt_tokens,
        completion_tokens,
    )


def urllib_transport(base_url: str, api_key: str, timeout: float):
    """Live transport. The key is added here and is not returned to the caller."""

    endpoint = validate_base_url(base_url) + "/chat/completions"

    def transport(payload: dict):
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/weight-space-experiment",
                "X-Title": "weight-space experiment",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"error": {"message": redact(raw)[:300]}}
            return exc.code, body

    return transport
