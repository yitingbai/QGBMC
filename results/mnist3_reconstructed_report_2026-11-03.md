# Reconstructed MNIST three-view extension experiment

## Version boundary

- This is a newly computed extension experiment. It does not recover the unavailable submission-time scripts.
- It must not replace or alter manuscript Tables 3–7, Figs. 2–5, or any untraceable O/M values.
- Suggested placement: new Table 8 plus a short protocol paragraph; all old reported numbers remain historical manuscript values.

## Material passport

- Status: `reconstructed_protocol_not_original_code`
- Samples: 60000; views: 3; dimensions: [4096, 1024, 2048]
- Construction: MNIST training images in fixed order; RGB conversion; resize 256; center crop 224; ImageNet normalization; ImageNet-pretrained VGG19, DenseNet121, and ResNet101 feature extractors.
- Published recipe anchor: https://openreview.net/references/attachment?id=z7vHla0Foa&name=supplementary_material
- Data SHA-256: `df3ef0ce9938bff419754110f37f8dee315fbdce996f19306e5373abc8044c68`
- Labels were not used for extraction or training; they were used only for post-training external clustering evaluation.

## Frozen protocol

- Seeds: 0, 1, 2; 50 epochs; Adam 1e-3; batch 256; hidden [256,128]; latent 64; MiniBatchKMeans with 3 initializations.
- Full variant: relative quality, gamma=2, tau=exp(-1), weighted SCA, per-feature reconstruction normalization.
- Uniform-quality ablation: identical protocol except quality is fixed to one for every view.
- Noise: element-wise Gaussian corruption of view 0 at rate 0.9; clean and noisy runs use matching seeds.

## Table 8 candidate: newly computed results (mean ± sample SD, n=3)

| Variant | Condition | ACC | NMI | ARI | PUR |
|---|---|---:|---:|---:|---:|
| full | clean | 0.7438 ± 0.0755 | 0.7648 ± 0.0480 | 0.6584 ± 0.0984 | 0.7936 ± 0.0605 |
| uniform_quality | clean | 0.6676 ± 0.1808 | 0.7381 ± 0.1067 | 0.5718 ± 0.2116 | 0.7407 ± 0.1377 |
| full | element0.9 | 0.8082 ± 0.0198 | 0.8172 ± 0.0148 | 0.7367 ± 0.0047 | 0.8443 ± 0.0090 |
| uniform_quality | element0.9 | 0.7793 ± 0.0104 | 0.7938 ± 0.0121 | 0.6944 ± 0.0281 | 0.8257 ± 0.0091 |

## Paired differences

- full_minus_uniform_clean:
  - ACC: +0.0762, 95% CI [-0.4464, +0.5987]
  - NMI: +0.0267, 95% CI [-0.2613, +0.3147]
  - ARI: +0.0867, 95% CI [-0.5253, +0.6986]
  - PUR: +0.0528, 95% CI [-0.3548, +0.4605]
- full_minus_uniform_element0.9:
  - ACC: +0.0290, 95% CI [-0.0341, +0.0921]
  - NMI: +0.0233, 95% CI [-0.0002, +0.0468]
  - ARI: +0.0423, 95% CI [-0.0221, +0.1067]
  - PUR: +0.0186, 95% CI [+0.0061, +0.0311]
- full_noise_minus_clean:
  - ACC: +0.0644, 95% CI [-0.0890, +0.2179]
  - NMI: +0.0524, 95% CI [-0.0977, +0.2024]
  - ARI: +0.0783, 95% CI [-0.1566, +0.3132]
  - PUR: +0.0507, 95% CI [-0.1152, +0.2166]

## Mechanism diagnostics for the full variant

- clean:
  - view 0: quality=0.9771, modulation=1.0097, decay occupancy=0.0097
  - view 1: quality=0.1414, modulation=0.1449, decay occupancy=0.9919
  - view 2: quality=0.7221, modulation=1.3371, decay occupancy=0.0067
- element0.9:
  - view 0: quality=0.1353, modulation=0.1353, decay occupancy=1.0000
  - view 1: quality=0.8821, modulation=1.1290, decay occupancy=0.0001
  - view 2: quality=0.9956, modulation=1.0046, decay occupancy=0.0000

## Evidence-bounded interpretation

- Under 0.9 corruption, the contaminated view 0 is assigned mean quality and modulation 0.1353 and enters the decay branch in 100% of batches. This directly verifies the intended gating response.
- Full minus uniform under strong noise is positive on all four mean metrics. The n=3 confidence intervals include zero for ACC, NMI (by a small margin), and ARI; only PUR excludes zero. With four exploratory metrics and three seeds, report this as a small, directionally consistent effect—not a general statistically established gain.
- The full variant scores higher under strong noise than under clean on average, with wide paired confidence intervals. A defensible hypothesis is that suppressing view 0 removes a harmful or redundant view; this experiment does not prove noise is beneficial.
- Clean runs have substantial seed sensitivity. Do not report a single seed or claim stable clean-data superiority.
- The reconstruction protocol is source-grounded but is not the original manuscript implementation. These values are suitable only as a separately labelled extension experiment.

## Audit trail

- Machine-readable summary: `mnist3_reconstructed_results_2026-11-03.json`
- Per-run table: `mnist3_reconstructed_runs_2026-11-03.csv`
- Every run contains `run.json`, `history.csv`, embeddings, preprocessing statistics/noise masks, the trained model, and the same data SHA-256.
