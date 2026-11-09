"""Probe client, cost cap, and mock mode. No network and no API key."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from wsp.mlp import parameter_count
from wsp.probe.client import HARD_CAP_EUR, ProbeClient, redact
from wsp.probe.pricing import ModelPrice, price_from_endpoints
from wsp.probe.prompt import messages_for, parse_readout
from wsp.probe.run import run_probe

ROOT = Path(__file__).resolve().parents[1]


def _price(**overrides) -> ModelPrice:
    data = dict(
        model_id="test/model",
        ceiling_prompt=0.0,
        ceiling_completion=1.0,
        cheap_prompt=0.0,
        cheap_completion=1.0,
        max_price_per_million={"prompt": 0.01, "completion": 1000000.01},
        temperature_supported=False,
        reasoning_efforts=("low",),
        source="pinned",
    )
    data.update(overrides)
    return ModelPrice(**data)


def _client(transport, tmp_path, cap=3.0):
    return ProbeClient(transport, tmp_path, cap, usd_to_eur_rate=1.0, sleeper=lambda _seconds: None)


def test_premium_tiers_cannot_set_the_price_ceiling():
    endpoints = [
        {"service_tier": "flex", "tag": "openai/flex", "pricing": {"prompt": "0.000005", "completion": "0.000025"}, "supported_parameters": ["reasoning"]},
        {"service_tier": None, "tag": "openai", "pricing": {"prompt": "0.00001", "completion": "0.00005"}, "supported_parameters": ["reasoning"]},
        {"service_tier": None, "tag": "amazon-bedrock/us-west-2", "pricing": {"prompt": "0.000011", "completion": "0.000055"}, "supported_parameters": ["reasoning"]},
        {"service_tier": "ultrafast", "tag": "openai/ultrafast", "pricing": {"prompt": "0.00006", "completion": "0.0003"}, "supported_parameters": ["reasoning"]},
        {"service_tier": "priority", "tag": "openai/fast", "pricing": {"prompt": "0.00002", "completion": "0.0001"}, "supported_parameters": ["temperature"]},
    ]
    price = price_from_endpoints("openai/gpt-6-astra", endpoints, 0.00001, 0.00005, ["low", "medium"], ["reasoning"])
    assert price is not None
    assert price.ceiling_prompt == pytest.approx(0.000011)
    assert price.ceiling_completion == pytest.approx(0.000055)
    assert price.cheap_prompt == pytest.approx(0.000005)
    assert price.temperature_supported is False
    assert price.max_price_per_million["completion"] < 60


def test_cap_stops_before_the_next_call(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": 2.0}}

    client = _client(transport, tmp_path, cap=3.0)
    price = _price()
    first = client.complete("test/model", [{"role": "user", "content": "one"}], 2, "low", None, price)
    second = client.complete("test/model", [{"role": "user", "content": "two"}], 2, "low", None, price)
    assert first.ok and first.attempted
    assert not second.ok and not second.attempted
    assert second.stop_reason == "cap"
    assert len(calls) == 1
    assert client.spent_eur == pytest.approx(2.0)


def test_three_4xx_stop_and_failures_are_not_cached(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 400, {"error": {"message": "bad request"}}

    client = _client(transport, tmp_path)
    price = _price(ceiling_completion=0.0)
    messages = [{"role": "user", "content": "hello"}]
    for _ in range(4):
        client.complete("test/model", messages, 8, "low", None, price)
    assert len(calls) == 3
    assert client.stop_reason == "4xx"
    cached = [path for path in tmp_path.glob("*.json") if path.name != "adaptations.json"]
    assert cached == []


def test_temperature_rejection_is_retried_once_and_then_cached(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        if "temperature" in payload:
            return 400, {"error": {"message": "Unsupported parameter: 'temperature' is not supported with this model."}}
        body = {"choices": [{"message": {"content": json.dumps({"label": "circle", "confidence": 0.5, "evidence": "raw"})}}], "usage": {"cost": 0.001, "prompt_tokens": 3, "completion_tokens": 4}}
        return 200, body

    client = _client(transport, tmp_path, cap=5.0)
    price = _price(temperature_supported=True, ceiling_completion=0.0001, cheap_completion=0.0001)
    messages = [{"role": "user", "content": "weights"}]
    first = client.complete("test/model", messages, 16, "low", 0.0, price)
    second = client.complete("test/model", messages, 16, "low", 0.0, price)
    assert first.ok and parse_readout(first.content)["label"] == "circle"
    assert second.cached and second.cost_eur == 0.0
    assert len(calls) == 2
    assert "temperature" not in calls[1]
    assert client.consecutive_4xx == 0
    assert "raw" not in (tmp_path / "adaptations.json").read_text()


def test_schema_rejection_falls_back_to_json_in_text(tmp_path):
    def transport(payload):
        if "response_format" in payload:
            return 400, {"error": {"message": "response_format is not supported"}}
        content = 'Here you go {"label": "square", "confidence": 0.25, "evidence": "drop me"}'
        return 200, {"choices": [{"message": {"content": content}}], "usage": {"cost": 0.01}}

    client = _client(transport, tmp_path)
    result = client.complete("test/model", [{"role": "user", "content": "w"}], 16, "low", None, _price(ceiling_completion=0.0001))
    parsed = parse_readout(result.content)
    assert parsed == {"label": "square", "confidence": 0.25}
    assert "evidence" not in parsed


def test_auth_failure_stops_immediately(tmp_path):
    calls = []

    def transport(payload):
        calls.append(1)
        return 401, {"error": {"message": "invalid key sk-or-v1-supersecretvalue"}}

    client = _client(transport, tmp_path)
    result = client.complete("test/model", [{"role": "user", "content": "w"}], 8, "low", None, _price(ceiling_completion=0.0))
    assert result.stop_reason == "auth"
    assert "sk-or-v1-supersecretvalue" not in (result.error or "")
    assert len(calls) == 1


def test_redact_and_cap_bounds():
    assert "supersecretvalue" not in redact("Bearer sk-or-v1-supersecretvalue")
    with pytest.raises(ValueError):
        ProbeClient(lambda payload: (200, {}), Path("/tmp"), cap_eur=HARD_CAP_EUR + 0.1, usd_to_eur_rate=1.0)


def test_prompts_differ_only_by_numbers_and_hide_the_condition():
    rng = np.random.default_rng(0)
    dim = parameter_count(2, 4)
    first = messages_for(rng.normal(size=dim), 2, 4)
    second = messages_for(rng.normal(size=dim), 2, 4)

    def skeleton(messages):
        return re.sub(r"-?\d+\.\d+", "#", messages[1]["content"])

    assert skeleton(first) == skeleton(second)
    text = first[1]["content"].lower()
    for leaked in ("shuffled", "permuted", "random", "shared", "independent", "truth"):
        assert leaked not in text
    for label in ("circle", "square", "triangle", "unrecognized"):
        assert label in text


def _tiny_probe_config(tmp_path):
    config = json.loads((ROOT / "configs" / "probe.json").read_text())
    config.update(
        {
            "hidden": 4,
            "grid": 8,
            "inr_steps": 8,
            "bank_per_class": 3,
            "cache_dir": str(tmp_path / "cache"),
        }
    )
    return config


def test_mock_mode_scores_without_a_key_and_rerun_is_free(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-should-not-appear")
    config = _tiny_probe_config(tmp_path)
    env = {"WSP_CACHE_DIR": str(tmp_path / "cache"), "WSP_USD_TO_EUR": "1"}

    def offline(_url):
        raise OSError("offline")

    first = run_probe(config, "mock", cap_eur=3.0, out_dir=tmp_path / "run", fetch=offline, env=env)
    second = run_probe(config, "mock", cap_eur=3.0, out_dir=tmp_path / "run", fetch=offline, env=env)
    blob = json.dumps(first)
    assert "should-not-appear" not in blob
    assert first["mode"] == "mock"
    assert first["spent_eur"] == 0.0
    assert first["metrics"]["openai/gpt-6-astra"]["accuracy"] == pytest.approx(14 / 15)
    assert first["metrics"]["anthropic/claude-opus-5.5"]["brier"] == pytest.approx((14 * 0.04 + 0.64) / 15)
    assert second["calls"]["cached"] == second["n_calls_planned"]
    assert second["calls"]["succeeded"] == 0
    written = (tmp_path / "run" / "summary.json").read_text()
    assert "evidence" not in written
    assert "mock" in written


def test_seed_configs_load_and_differ_only_by_seed():
    from wsp.probe.run import load_probe_config

    loaded = [load_probe_config(ROOT / f"configs/probe_s{seed}_p1.json") for seed in (1, 2, 3)]
    assert [config["seed"] for config in loaded] == [1, 2, 3]
    without_seed = []
    for config in loaded:
        copy = dict(config)
        copy.pop("seed")
        without_seed.append(copy)
    assert without_seed[0] == without_seed[1] == without_seed[2]
    assert load_probe_config(ROOT / "configs/probe.json")["seed"] == 0


def test_default_probe_reserve_stays_under_three_euros():
    config = json.loads((ROOT / "configs" / "probe.json").read_text())

    def offline(_url):
        raise OSError("offline")

    summary = run_probe(config, "dry-run", cap_eur=3.0, fetch=offline, env={"WSP_USD_TO_EUR": "0.89158"})
    assert summary["pricing_source"] == "pinned"
    assert summary["plan"]["worst_case_eur"] < 3.0
    assert summary["plan"]["expected_eur"] < summary["plan"]["worst_case_eur"]
    assert summary["n_calls_planned"] == 30
    assert summary["baselines"]["uniform_chance"] == pytest.approx(0.25)
    for row in summary["plan"]["by_model"].values():
        assert row["temperature_supported"] is False
