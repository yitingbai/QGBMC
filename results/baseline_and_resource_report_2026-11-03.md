## Material Passport

- Experiment ID: `revision_table9_table11_caltech_scene_5seeds`
- Type: runnable external-baseline and resource comparison
- Status: `REPRODUCED` within the recorded two-view protocols
- Seeds: 0, 1, 2, 3, 4
- Datasets: Caltech101-20 and Scene-15, using the same frozen two views for every method within each dataset
- Boundary: newly computed revision evidence; does not replace manuscript Tables 3--7 or recover their O/M cells

# Added runnable baselines and resource measurements

## Table 9 candidate: clustering results

Mean ± sample SD over five seeds. All values are on a 0--1 scale.

| Dataset | Method | ACC | NMI | ARI |
|---|---|---:|---:|---:|
| Caltech101-20 | Concat-KMeans | 0.4535 ± 0.0237 | 0.6348 ± 0.0131 | 0.3496 ± 0.0243 |
| Caltech101-20 | DealMVC | 0.3666 ± 0.0113 | 0.5656 ± 0.0091 | 0.2928 ± 0.0174 |
| Caltech101-20 | COMPLETER | **0.7121 ± 0.0302** | **0.6955 ± 0.0167** | **0.7792 ± 0.0635** |
| Caltech101-20 | QGBMC v3 | 0.5064 ± 0.0141 | 0.6650 ± 0.0087 | 0.4030 ± 0.0143 |
| Scene-15 | Concat-KMeans | 0.3701 ± 0.0030 | 0.4036 ± 0.0026 | 0.2141 ± 0.0022 |
| Scene-15 | DealMVC | 0.3856 ± 0.0111 | 0.4183 ± 0.0068 | 0.2407 ± 0.0076 |
| Scene-15 | COMPLETER | 0.4046 ± 0.0114 | 0.4345 ± 0.0197 | 0.2330 ± 0.0282 |
| Scene-15 | QGBMC v3 | **0.4319 ± 0.0128** | **0.4455 ± 0.0045** | **0.2636 ± 0.0097** |

### Protocol notes

- QGBMC v3 uses relative quality, gamma=2, tau=exp(-1), weighted SCA, per-feature reconstruction normalization, 50 epochs, and batch 256.
- COMPLETER uses its official implementation at commit `624dd9fd10431802214fab1d7b34c44f5a0a92df`, `missing_rate=0`, 500 epochs, and batch 256.
- DealMVC uses its official network at commit `c7301f18b8b793f23e1f5565f15c75a8a937961d`. The official repository exposes only BBCSport, so the transparent adapter replaces the data entry while retaining the official 300+100+50 training stages and defaults.
- Concat-KMeans applies the same per-view z-score preprocessing used by QGBMC, concatenates the two views, and uses KMeans with `n_init=20`.
- COMPLETER's official adapter returns ACC, NMI, and ARI but not PUR. Table 9 therefore uses the common three-metric intersection rather than fabricating a missing value.

## Table 11 candidate: measured runtime and peak memory

These values are end-to-end measurements of each method's recorded protocol. Epoch budgets differ by method and the times must not be interpreted as normalized architectural speed.

| Dataset | Method | Budget | Mean time s | Peak reserved MiB | Batch / constraint | Status |
|---|---|---|---:|---:|---|---|
| Caltech101-20 | Concat-KMeans | KMeans n_init=20 | 10.7 ± 0.1 | not captured (CPU) | full data | success 5/5 |
| Caltech101-20 | DealMVC | 300+100+50 epochs | 70.7 ± 15.7 | 522.0 | batch 256; attention dimension is batch-coupled | success 5/5 |
| Caltech101-20 | COMPLETER | 500 epochs | 43.2 ± 2.5 | 422.0 | batch 256 | success 5/5 |
| Caltech101-20 | QGBMC v3 | 50 epochs | 12.0 ± 0.4 | 118.0 | batch 256 | success 5/5 |
| Scene-15 | Concat-KMeans | KMeans n_init=20 | 1.9 ± 0.0 | not captured (CPU) | full data | success 5/5 |
| Scene-15 | DealMVC | 300+100+50 epochs | 127.5 ± 1.3 | 660.0 | batch 256; attention dimension is batch-coupled | success 5/5 |
| Scene-15 | COMPLETER | 500 epochs | 97.4 ± 36.0 | 246.0 | batch 256 | success 5/5 |
| Scene-15 | QGBMC v3 | 50 epochs | 17.2 ± 0.3 | 74.0 | batch 256 | success 5/5 |

## Evidence-bounded interpretation

1. QGBMC v3 does not consistently outperform the runnable baselines. It is substantially below COMPLETER on Caltech101-20 and has the highest means on Scene-15.
2. The two-dataset result supports a mixed, dataset-dependent comparison and directly contradicts a universal state-of-the-art superiority claim.
3. Every newly added method completed at batch 256 on the recorded RTX 4090 environment. Thus the new comparison table contains no unexplained O/M cells.
4. DealMVC's official attention implementation couples its hidden dimension to batch size 256. Reducing only the batch size would change the network architecture, so a 256→128→64 fallback is not reported as a like-for-like DealMVC test.
5. The successful new resource audit does not reconstruct the causes of the legacy O/M entries in manuscript Tables 3--4. Those legacy cells remain historical and must not be presented as newly verified.
6. Runtime comparisons are descriptive because training budgets and objectives differ across methods.

## Audit trail

- QGBMC v3 runs: `runs/{caltech101_20_completer2,scene15_completer2}__full__clean__v3__e50__s{0..4}`
- DealMVC runs: `baseline_runs/dealmvc_{caltech,scene}_official_e300_100_50_s{0..4}`
- COMPLETER summaries: `baseline_results/completer_{caltech,scene}_complete_5seeds.json`
- Concat-KMeans: `baseline_runs/concat_kmeans_{caltech101_20_completer2,scene15_completer2}_s{0..4}.json`
- DealMVC smoke output is engineering-only and excluded from the table.
