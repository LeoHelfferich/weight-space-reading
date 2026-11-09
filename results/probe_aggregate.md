# Probe aggregates

The paid probe ran on the owner's machine, outside this repository. This file keeps counts only. Raw model responses, prompts, and weight dumps are not committed. No API key is recorded here.

Model ids: `openai/gpt-6-astra`, `anthropic/claude-opus-5.5`.

## 1. Local experiment (no API)

`python -m wsp.local_experiment --config configs/local.json` took 75 s, 19/19 tests green. All 273 numeric values bit-identical to the committed results/local_summary.json (only platform/Python version and image paths differ).

| 2D linear probe (ridge) on weights | value |
| --- | --- |
| shared init, raw | 0.889 |
| shared init, neurons permuted | 0.389 |
| shared init, after neuron sort | 0.833 (also on permuted: 0.833) |
| shared init, shuffled within tensors | 0.500 |
| independent init, raw | 0.444 |
| independent init, after sort | 0.500 |
| 3D shared init raw / sorted | 0.778 / 0.889 |
| 3D independent init raw / sorted | 0.500 / 0.389 |

Permuted networks still render the same shape (IoU to own target 0.999).

## 2. Paid pilot (configs/probe.json)

15 networks (3 per condition), 2 models, 30 calls, 30 successful, 0 errors. Weights as numbers in the prompt (~2,400-2,800 tokens); answer is label + confidence; prompt forbids recomputing the forward pass. Cost 0.695 EUR total (cap 3 EUR), Astra 0.519 EUR, Opus 5.5 0.176 EUR. Seed 0.

| Accuracy (n=3 each) | GPT-6 Astra | Claude Opus 5.5 | centroid raw | centroid sorted | run the net |
| --- | --- | --- | --- | --- | --- |
| shared init | 3/3 | 2/3 | 0.33 | 0.33 | 1.00 |
| independent init | 3/3 | 2/3 | 0.00 | 0.67 | 1.00 |
| neurons permuted | 3/3 | 2/3 | 0.33 | 0.33 | 1.00 |
| shuffled (expected "unrecognized") | 0/3 | 0/3 | 0.00 | 1.00 | 1.00 |
| random weights (expected "unrecognized") | 3/3 | 3/3 | 0.00 | 0.00 | 1.00 |
| total (15) | 0.80 | 0.60 | 0.13 | 0.47 | 1.00 |

Other baselines: always "unrecognized" 0.40, chance 0.25.

Calibration: Astra Brier 0.112, ECE 0.197; Opus 5.5 Brier 0.177, ECE 0.057 (12 of 15 answers at 0.40-0.60).

## Section 3, final run

60 networks, 12 per condition. 120 calls. Cost 2.71 EUR. One answer per network per model. Each network has 337 numbers.

The final run is the pooled 60 = pilot 15 + seeds 1-3 (45 new). Cost 2.709 EUR total (seeds 2.014 EUR). Configs: `configs/probe.json` (seed 0), `configs/probe_s1_p1.json` (seed 1), `configs/probe_s2_p1.json` (seed 2), `configs/probe_s3_p1.json` (seed 3).

Correct answers out of 12. "Code" sorts neurons by a fixed rule, then compares with 42 known networks. "Run the net" is the forward pass.

| Condition | Astra | Opus 5.5 | Code | Run the net |
| --- | ---: | ---: | ---: | ---: |
| Real, shared init | 12 | 8 | 7 | 12 |
| Real, independent init | 11 | 8 | 7 | 12 |
| Permuted | 12 | 6 | 7 | 12 |
| Shuffled | 1 | 0 | 12 | 11 |
| Random | 12 | 12 | 0 | 12 |
| Total of 60 | 48 | 34 | 33 | 59 |

Always answering "no shape" scores 24 of 60.

On the 12 shuffled networks, Astra still named a shape in 11 of 12 cases, 7 of them a circle.

Claude Opus 5.5 did not say "square" in any of its 60 answers.

Calibration: Astra Brier 0.114, ECE 0.144 (pooled 60); Opus 5.5 Brier 0.181, ECE 0.110 (pooled 60); Opus gave 47 of 60 answers at 0.40-0.60.
