# QGBMC reproducibility archive

This archive contains the QGBMC implementation, controlled experiment harness, configurations, aggregation tools, and measured result summaries used in the manuscript and Supplementary Information.

## Scope

- `qgbmc/` contains data I/O, network, quality estimation, losses, training, and evaluation modules.
- `configs/` contains frozen default, ablation, noise, sensitivity, threshold, and smoke configurations.
- `train.py`, `evaluate.py`, and `scripts/` provide the principal execution and aggregation entry points.
- `run_completer_complete.py`, `run_dealmvc_complete.py`, and `run_concat_kmeans.py` implement the matched-view comparison wrappers. Third-party baseline repositories are not redistributed; obtain them from their official sources and pass their local paths through the documented command-line options.
- `results/` contains the measured experiment summaries, factorial-ablation outputs, resource report, data contract, and experiment design.
- `baseline_results/` contains the retained matched-view COMPLETER summaries.
- `run_inventory.csv` is a sanitized inventory of completed local runs. Raw checkpoints, embeddings, predictions, corruption masks, and public benchmark data are excluded from this compact code archive because the local run directory is approximately 3.6 GB. They may be deposited as a separate Zenodo data component if desired.
- `ARTIFACT_READINESS.md` records the completed package checks and the two repository-level actions that remain before DOI release.

## Verified environment

- Python 3.9
- PyTorch 2.1.2 with CUDA 12.1
- NumPy 1.26.4
- SciPy 1.13.1
- scikit-learn 1.6.1
- NVIDIA RTX 4090 24 GB for the reported controlled runs

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Data preparation

The archive does not redistribute public benchmark data. Follow `results/data_contract.md` and use the conversion utilities in `scripts/` or `build_reconstructed_image_views.py`. Verify the resulting data hashes against the contract before running the reported configurations.

## Main QGBMC run

```bash
python train.py --config config.yaml --dataset handwritten2 --variant full --seed 0 --epochs 50 --quality-mode relative --reconstruction-normalization per_feature_mean --gamma 2 --tau 0.36787944117144233 --lambda-sca 1 --sca-mode weighted
```

The principal variants are:

- `full`: QGM and quality-weighted SCA enabled.
- `uniform_quality`: quality values fixed to one.
- `shuffled_quality`: observed quality values assigned to the wrong views.
- `wo_sca`: QGM-only factorial cell with ordinary InfoNCE.
- `wo_qgm`: SCA-only factorial cell without gradient modulation.

Aggregate completed runs with:

```bash
python evaluate.py --runs PATH_TO_RUNS --output results.json
```

Run the test suite with:

```bash
python -m pytest -q
```

## Repository release checklist

Before public Zenodo release, the authors should select an explicit software license, add the final DOI to the manuscript and response letter, and decide whether the 3.6 GB raw-run archive will be uploaded as a separate linked record.
