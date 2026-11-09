# Weight-space reading

An independent, small replication of one claim in [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL, November 2026).

The post reports that a frontier model can tell what a network does by reading its weights. This repository trains its own tiny shape networks on synthetic shapes and local compute, then asks two current models the same kind of question. It is not affiliated with the authors or with MIT. It does not reuse their logs, weights, or datasets.

## Networks, conditions, baselines

Each scored network is an implicit MLP, `2 → 16 → 16 → 1` (**337 weights**, ReLU). It maps `(x, y)` in `[-1, 1]` to the probability that the point is inside one filled shape: circle, square, or triangle. The same local command also trains a 3D occupancy MLP (sphere, cube, octahedron, 353 weights). The paid probe uses the 2D network only.

Conditions:

- **Shared init.** Every network in the run starts from the same draw.
- **Independent init.** Each network has its own draw.
- **Permuted.** Hidden units of a trained network are reordered. The function is unchanged.
- **Shuffled.** Entries are shuffled inside each weight matrix.
- **Random.** A fresh initialization that is never trained.

Local baselines, with no API:

- Nearest centroid on the raw weights, and the same centroid after hidden units are sorted by incoming-weight norm.
- A ridge linear readout on the flat weights.
- Running the network and scoring the picture (intersection-over-union; 0.5 counts as a match).
- Always answering `unrecognized`.

The LLM probe sends the weights, rounded to 4 decimals, to `openai/gpt-6-astra` and `anthropic/claude-opus-5.5` on OpenRouter. Each model is asked once per network which shape the weights encode, or `unrecognized`. Ids are set in `configs/probe.json`.

## Reproduce

Local training needs no key and makes no network calls:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
pytest
python -m wsp.local_experiment --config configs/local.json --out results
```

The paid probe reads `OPENROUTER_API_KEY` from the environment. Set the variable in the environment; do not put it on the command line. `--cap-eur` defaults to 3 and cannot be set above 5. Successful replies are cached in `results/probe_cache` (gitignored), so a rerun of the same request costs nothing. Failed calls are not cached.

```bash
python -m wsp.probe --config configs/probe.json --dry-run --cap-eur 3
python -m wsp.probe --config configs/probe.json --mock --cap-eur 3
python -m wsp.probe --config configs/probe.json --cap-eur 3
```

`--dry-run` and `--mock` need no key. Cost, FX, and the other variable names are in [REPRODUCE.md](REPRODUCE.md).

## Results

The paid probe ran on the owner's machine, not in this repository. Counts only are in [results/probe_aggregate.md](results/probe_aggregate.md). Raw responses are deliberately not committed.

Final run (section 3): 60 networks, 12 per condition, 120 calls, 2.71 EUR. The pooled 60 = pilot 15 + seeds 1-3 (45 new). Cost 2.709 EUR total (seeds 2.014 EUR). Models: `openai/gpt-6-astra` and `anthropic/claude-opus-5.5`. Configs: `configs/probe.json` (seed 0), `configs/probe_s1_p1.json`, `configs/probe_s2_p1.json`, `configs/probe_s3_p1.json`. Calibration, pooled 60: Astra Brier 0.114, ECE 0.144; Opus 5.5 Brier 0.181, ECE 0.110; Opus gave 47 of 60 answers at 0.40-0.60.

Section 2 is the 15-network pilot (`configs/probe.json`, seed 0): 30 calls, 30 successful, 0 errors, cost 0.695 EUR total (Astra 0.519 EUR, Opus 5.5 0.176 EUR). Totals: Astra 0.80, Opus 5.5 0.60, centroid raw 0.13, centroid sorted 0.47, run the net 1.00. Always "unrecognized" 0.40, chance 0.25. Calibration: Astra Brier 0.112, ECE 0.197; Opus 5.5 Brier 0.177, ECE 0.057 (12 of 15 answers at 0.40-0.60).

Correct out of 12:

| Condition | Astra | Opus 5.5 | Code | Run the net |
| --- | ---: | ---: | ---: | ---: |
| Real, shared init | 12 | 8 | 7 | 12 |
| Real, independent init | 11 | 8 | 7 | 12 |
| Permuted | 12 | 6 | 7 | 12 |
| Shuffled | 1 | 0 | 12 | 11 |
| Random | 12 | 12 | 0 | 12 |
| Total of 60 | 48 | 34 | 33 | 59 |

Always answering "no shape" scores 24 of 60. On the shuffled networks, Astra still named a shape in 11 of 12 cases, 7 of them a circle. Claude Opus 5.5 did not say "square" in any of its 60 answers.

Local training aggregates, separate from that probe, are in `results/local_summary.json` and `results/figures/`.

Chance for a 3-way class guess is 0.333.

**2D, shared init** (108 train, 36 test). Test IoU 0.999.

| | Trained | Permuted | Shuffled | Random |
| --- | ---: | ---: | ---: | ---: |
| IoU vs own target | 0.999 | 0.999 | 0.415 | 0.094 |
| Ridge readout | 0.889 | 0.389 | 0.500 | 0.000 |
| Nearest centroid | 0.889 | 0.278 | 0.222 | 0.000 |

Sorting hidden units and refitting the ridge readout scores 0.833 on the plain test set and 0.833 on the permuted set. PCA diffusion samples (n=48) are a recognizable shape every time (mean best IoU 0.76) and match the requested class 0.52 of the time.

**2D, independent init.** Test IoU is 0.999. The ridge readout is 0.444. Samples match the requested class 0.15 of the time.

**3D, shared init** (54 train, 18 test). Test IoU 0.997. Ridge 0.778, and 0.889 after the neuron sort. Sample class accuracy 0.83 (n=24). Independent init: ridge 0.500, sample class accuracy 0.08.

## Limitations

The networks are tiny, and the probe uses 2D shapes. Each network is sent to each model once. The counts above are that one pass. Raw replies stay on the machine that ran the probe.

## Citation

```bibtex
@misc{takehi2026weightspace,
  title        = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}},
  note         = {Blog post}
}
```

The code in this repository is MIT. The blog post remains the authors'.
