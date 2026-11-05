## Material Passport

- Experiment ID: `aloi_1k_full_clean_v3_e20_seeds0_1_2`
- Type: large-scale multi-view clustering execution
- Status: `REPRODUCED` for the frozen v3 configuration
- Dataset: ALOI_1K, 110,250 samples, 4 views, 1,000 clusters
- Data SHA-256: `a30c1b1b390e7e458c9444d2815109eeb27a9294c5eb8540b6dda7a91b8328df`
- Protocol boundary: newly computed revision experiment; not an original-paper result and not YTF50

# ALOI_1K v3 three-seed scale experiment

## Frozen configuration

- Seeds: 0, 1, 2
- Epochs: 20
- Batch size: 256
- Hidden layers: [256, 128]
- Latent dimension: 64
- Adam learning rate: 0.001
- Relative quality: gamma=2, tau=exp(-1)
- Weighted SCA, lambda=1
- Per-feature reconstruction normalization
- MiniBatchKMeans, n_init=3

## Per-seed results

| Seed | ACC | NMI | ARI | PUR | Training s | Clustering s | Peak reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4339 | 0.7760 | 0.3139 | 0.4765 | 188.6 | 20.5 | 84.0 |
| 1 | 0.4240 | 0.7718 | 0.2939 | 0.4742 | 176.4 | 13.3 | 84.0 |
| 2 | 0.4145 | 0.7715 | 0.2942 | 0.4645 | 179.8 | 12.8 | 84.0 |

## Three-seed summary

Mean ± sample SD:

- ACC: 0.4241 ± 0.0097
- NMI: 0.7731 ± 0.0025
- ARI: 0.3007 ± 0.0115
- PUR: 0.4717 ± 0.0063
- Training time: 181.6 ± 6.3 s
- Clustering time: 15.5 ± 4.3 s
- Peak CUDA allocated: 81.2 MiB
- Peak CUDA reserved: 84.0 MiB

## Evidence-bounded interpretation

1. The frozen v3 implementation completed all three runs on 110,250 samples without OOM, crash, or non-finite values.
2. The experiment supports a ten-to-the-fifth-scale feasibility and resource claim under the recorded batch size and hardware.
3. It does not establish state-of-the-art large-scale clustering performance because no matched large-scale external baseline has yet been completed.
4. ALOI-1K and YTF50 have different data characteristics. This experiment supports execution feasibility on ALOI-1K and should not be interpreted as a YTF50 result.
5. The older ALOI seed-0 development run used tau=0.1 and must not be pooled with these v3 runs.

## Audit trail

Each run directory contains `run.json`, `history.csv`, preprocessing output, embeddings and predictions, model state, configuration, timing, peak CUDA memory, and the input data hash:

- `runs/aloi_1k__full__clean__v3__e20__s0`
- `runs/aloi_1k__full__clean__v3__e20__s1`
- `runs/aloi_1k__full__clean__v3__e20__s2`
