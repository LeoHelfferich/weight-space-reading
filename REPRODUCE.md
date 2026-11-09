# Reproduce

The committed local numbers were produced on Linux x86_64, CPython 3.12.3, NumPy 2.5.4, with seed 0. A one-minute run on a 4-core CPU is the right expectation for the local command. Bit-identical floats on another OS are not guaranteed; the qualitative gaps below are large enough that small numeric drift will not hide them.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```

## Environment variables

Names only. Do not write values into the repo, shell history you commit, or log files.

| Name | When |
| --- | --- |
| `OPENROUTER_API_KEY` | Live probe only. Absent in `--dry-run` and `--mock`. |
| `OPENROUTER_BASE_URL` | Optional. Must be `https://openrouter.ai/...`. Default `https://openrouter.ai/api/v1`. |
| `WSP_USD_TO_EUR` | Optional FX override (EUR per 1 USD). Default is a live Frankfurter rate, then the pinned rate in `configs/probe.json`. |
| `WSP_CACHE_DIR` | Optional response-cache directory. Default `results/probe_cache`, which is gitignored. |

No other secret is read. The local command makes no network calls.

## Local experiment

```bash
python -m wsp.local_experiment --config configs/local.json --out results
```

About one minute here. Writes `results/local_summary.json` and `results/figures/`. Pre-registered decision threshold: a render matches a shape when intersection-over-union is at least 0.5.

## Paid probe

Dry-run (no key, no chat request). It trains the small probe bank so the token estimate is measured, then reads public OpenRouter prices when the network is up:

```bash
python -m wsp.probe --config configs/probe.json --dry-run --cap-eur 3
```

Mock (no key, scripted answers, checks scoring, cache, and the cost summary):

```bash
python -m wsp.probe --config configs/probe.json --mock --cap-eur 3
```

Live, after `OPENROUTER_API_KEY` is set in the environment (not on the command line):

```bash
python -m wsp.probe --config configs/probe.json --cap-eur 3
```

The probe that produced the published counts ran on the owner's machine, outside this repository. Raw responses are deliberately not committed. Counts are in `results/probe_aggregate.md`.

That run used `configs/probe.json` (seed 0, 15 networks) and `configs/probe_s1_p1.json`, `configs/probe_s2_p1.json`, `configs/probe_s3_p1.json` (seeds 1, 2, and 3). Final run, section 3: 60 networks, 12 per condition, 120 calls, 2.71 EUR. The pooled 60 = pilot 15 + seeds 1-3 (45 new). Cost 2.709 EUR total (seeds 2.014 EUR). Section 2, the 15-network pilot: 30 calls, 30 successful, 0 errors, 0.695 EUR total (Astra 0.519 EUR, Opus 5.5 0.176 EUR). Models: `openai/gpt-6-astra` and `anthropic/claude-opus-5.5`.

`--cap-eur` defaults to 3 and cannot be set above 5. The client refuses a call whose ceiling estimate would cross the cap. Neither model lists `temperature` in `supported_parameters`, so the probe does not send it. A 400 that names temperature is retried once without it and that fact is remembered for the rest of the run. A cached rerun adds nothing. Failed calls are not cached.
