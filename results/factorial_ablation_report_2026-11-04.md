## Material Passport

- Origin Skill: `experiment-agent`
- Origin Mode: `run + validate`
- Origin Date: 2026-11-04
- Verification Status: `REPRODUCED` for the frozen v3 protocol
- Version Label: `qgbmc_v3_factorial_ablation_v1`
- New formal runs: 40
- Total paired runs analysed: 80, including existing Full and Base runs
- Environment: Python 3.9.21; PyTorch 2.1.2+cu121; NVIDIA RTX 4090

# QGBMC v3 minimal 2 x 2 factorial ablation

## Research question

Does the clustering behaviour of QGBMC arise from quality-conditioned gradient routing, quality-weighted Selective Contrastive Alignment (SCA), or their interaction?

## Frozen protocol

- Datasets: HandWritten2 and Hdigit.
- Conditions: clean and view-0 element-wise Gaussian replacement at rate 0.9.
- Seeds: 0--4, paired across variants.
- Training: 50 epochs, batch 256, Adam 1e-3, hidden [256,128], latent 64.
- Quality: relative entropy, gamma=2, tau=exp(-1).
- Reconstruction: per-feature mean.
- SCA: weighted InfoNCE when enabled; lambda=1; temperature=0.5.
- Diagnostics: branch occupancy and raw/modulated alignment-gradient RMS.

## Factorial variants

| Display name | Code variant | Gradient routing | Quality-weighted SCA |
|---|---|---:|---:|
| Base | `uniform_quality` | off | off |
| Routing-only | `wo_sca` | on | off |
| Alignment-only | `wo_qgm` | off | on |
| Full | `full` | on | on |

## Five-seed results

Mean +/- sample standard deviation.

| Dataset | Condition | Variant | ACC | NMI | ARI | PUR |
|---|---|---|---:|---:|---:|---:|
| HandWritten2 | clean | Base | 0.8027 +/- 0.0246 | 0.7836 +/- 0.0220 | 0.7001 +/- 0.0273 | 0.8090 +/- 0.0173 |
| HandWritten2 | clean | Routing-only | 0.8224 +/- 0.0146 | 0.7965 +/- 0.0199 | 0.7209 +/- 0.0284 | 0.8224 +/- 0.0146 |
| HandWritten2 | clean | Alignment-only | 0.8156 +/- 0.0249 | 0.7933 +/- 0.0227 | 0.7247 +/- 0.0301 | 0.8208 +/- 0.0175 |
| HandWritten2 | clean | Full | 0.8009 +/- 0.0022 | 0.7597 +/- 0.0067 | 0.6794 +/- 0.0055 | 0.8009 +/- 0.0022 |
| HandWritten2 | element 0.9 | Base | 0.5264 +/- 0.0363 | 0.4770 +/- 0.0286 | 0.3471 +/- 0.0472 | 0.5525 +/- 0.0366 |
| HandWritten2 | element 0.9 | Routing-only | 0.5250 +/- 0.0455 | 0.4777 +/- 0.0194 | 0.3462 +/- 0.0372 | 0.5503 +/- 0.0329 |
| HandWritten2 | element 0.9 | Alignment-only | 0.5440 +/- 0.0438 | 0.4886 +/- 0.0156 | 0.3544 +/- 0.0359 | 0.5628 +/- 0.0342 |
| HandWritten2 | element 0.9 | Full | 0.5412 +/- 0.0404 | 0.4901 +/- 0.0187 | 0.3658 +/- 0.0308 | 0.5606 +/- 0.0382 |
| Hdigit | clean | Base | 0.9941 +/- 0.0011 | 0.9835 +/- 0.0022 | 0.9870 +/- 0.0024 | 0.9941 +/- 0.0011 |
| Hdigit | clean | Routing-only | 0.9286 +/- 0.0442 | 0.9260 +/- 0.0188 | 0.8952 +/- 0.0500 | 0.9286 +/- 0.0442 |
| Hdigit | clean | Alignment-only | 0.9953 +/- 0.0003 | 0.9859 +/- 0.0008 | 0.9896 +/- 0.0007 | 0.9953 +/- 0.0003 |
| Hdigit | clean | Full | 0.8841 +/- 0.0888 | 0.9004 +/- 0.0328 | 0.8533 +/- 0.0868 | 0.8850 +/- 0.0877 |
| Hdigit | element 0.9 | Base | 0.5750 +/- 0.0435 | 0.5142 +/- 0.0229 | 0.3975 +/- 0.0319 | 0.5854 +/- 0.0400 |
| Hdigit | element 0.9 | Routing-only | 0.5714 +/- 0.0419 | 0.5106 +/- 0.0299 | 0.3925 +/- 0.0357 | 0.5847 +/- 0.0366 |
| Hdigit | element 0.9 | Alignment-only | 0.5535 +/- 0.0236 | 0.5183 +/- 0.0177 | 0.3957 +/- 0.0213 | 0.5776 +/- 0.0306 |
| Hdigit | element 0.9 | Full | 0.5720 +/- 0.0289 | 0.5287 +/- 0.0172 | 0.4097 +/- 0.0288 | 0.5944 +/- 0.0240 |

## Paired ACC contrasts

Confidence intervals are paired seed-wise 95% t intervals with n=5. P values are descriptive and are not multiplicity-corrected.

| Dataset | Condition | Contrast | Mean difference | 95% CI | p |
|---|---|---|---:|---:|---:|
| HandWritten2 | clean | Routing-only - Base | +0.0197 | [-0.0175, +0.0569] | 0.2159 |
| HandWritten2 | clean | Alignment-only - Base | +0.0129 | [-0.0207, +0.0465] | 0.3468 |
| HandWritten2 | clean | Full - Routing-only | -0.0215 | [-0.0384, -0.0046] | 0.0241 |
| HandWritten2 | clean | Full - Alignment-only | -0.0147 | [-0.0445, +0.0151] | 0.2426 |
| HandWritten2 | clean | Factorial interaction | -0.0344 | [-0.0786, +0.0098] | 0.0970 |
| HandWritten2 | element 0.9 | Routing-only - Base | -0.0014 | [-0.0304, +0.0276] | 0.8997 |
| HandWritten2 | element 0.9 | Alignment-only - Base | +0.0176 | [+0.0036, +0.0316] | 0.0249 |
| HandWritten2 | element 0.9 | Full - Routing-only | +0.0162 | [-0.0242, +0.0566] | 0.3281 |
| HandWritten2 | element 0.9 | Full - Alignment-only | -0.0028 | [-0.0226, +0.0170] | 0.7144 |
| HandWritten2 | element 0.9 | Factorial interaction | -0.0014 | [-0.0339, +0.0311] | 0.9106 |
| Hdigit | clean | Routing-only - Base | -0.0655 | [-0.1204, -0.0106] | 0.0296 |
| Hdigit | clean | Alignment-only - Base | +0.0012 | [-0.0002, +0.0026] | 0.0758 |
| Hdigit | clean | Full - Routing-only | -0.0445 | [-0.1407, +0.0517] | 0.2682 |
| Hdigit | clean | Full - Alignment-only | -0.1112 | [-0.2212, -0.0012] | 0.0485 |
| Hdigit | clean | Factorial interaction | -0.0457 | [-0.1420, +0.0506] | 0.2581 |
| Hdigit | element 0.9 | Routing-only - Base | -0.0037 | [-0.0133, +0.0059] | 0.3496 |
| Hdigit | element 0.9 | Alignment-only - Base | -0.0215 | [-0.0527, +0.0096] | 0.1270 |
| Hdigit | element 0.9 | Full - Routing-only | +0.0006 | [-0.0715, +0.0727] | 0.9832 |
| Hdigit | element 0.9 | Full - Alignment-only | +0.0185 | [-0.0312, +0.0681] | 0.3601 |
| Hdigit | element 0.9 | Factorial interaction | +0.0221 | [-0.0287, +0.0730] | 0.2936 |

## Mechanism diagnostics from existing Full runs

Values aggregate the final 10 epochs across five seeds. `Epoch ranking` is the fraction of seed-epoch summaries in which the designated view 0 had lower quality than view 1; it is not a batch-level AUROC.

| Dataset | Rate | q0 | q1 | q1-q0 | Decay occ. v0 | Decay occ. v1 | Grad ratio v0 | Epoch ranking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HandWritten2 | 0.0 | 1.0000 | 0.1353 | -0.8647 | 0.0000 | 1.0000 | 1.0000 | 0.00 |
| HandWritten2 | 0.3 | 0.1681 | 0.9672 | 0.7991 | 0.9621 | 0.0379 | 0.1673 | 1.00 |
| HandWritten2 | 0.6 | 0.2510 | 0.8843 | 0.6333 | 0.8662 | 0.1338 | 0.2494 | 0.94 |
| HandWritten2 | 0.9 | 0.1353 | 1.0000 | 0.8647 | 1.0000 | 0.0000 | 0.1353 | 1.00 |
| Hdigit | 0.0 | 0.1354 | 0.9999 | 0.8646 | 0.9999 | 0.0001 | 0.1361 | 1.00 |
| Hdigit | 0.3 | 0.1354 | 0.9999 | 0.8646 | 0.9999 | 0.0001 | 0.1361 | 1.00 |
| Hdigit | 0.6 | 0.1353 | 1.0000 | 0.8647 | 1.0000 | 0.0000 | 0.1353 | 1.00 |
| Hdigit | 0.9 | 0.1353 | 1.0000 | 0.8647 | 1.0000 | 0.0000 | 0.1353 | 1.00 |

## Evidence-bounded interpretation

1. No dataset-condition shows a factorial interaction whose 95% CI excludes zero. The present evidence does not support a positive performance synergy between QGM routing and quality-weighted SCA.
2. On clean HandWritten2, Full was lower than Routing-only by 0.0215 ACC. On clean Hdigit, routing reduced ACC relative to Base by 0.0655, and Full was lower than Alignment-only by 0.1112. These results identify gradient routing as the main source of the clean-data degradation in Hdigit.
3. Under HandWritten2 rate-0.9 corruption, Alignment-only exceeded Base by 0.0176 ACC, while Full and Alignment-only were nearly identical. The measurable gradient attenuation did not add a stable clustering gain beyond quality-weighted SCA.
4. Under Hdigit rate-0.9 corruption, all four ACC means were close and all primary paired intervals included zero. Full had the highest NMI, ARI and PUR means, but this is descriptive secondary evidence.
5. The topology proxy reverses its view ranking on HandWritten2 when corruption is introduced, but it assigns view 0 the minimum quality on clean Hdigit as well as on corrupted Hdigit. This directly explains why routing can be correct as an implemented operation yet harmful as a data-dependent decision.

## Statistical validation and fallacy scan

- Overall confidence: `CAUTION` because n=5 yields wide intervals and several exploratory contrasts were inspected.
- Primary descriptive target: ACC and the factorial interaction within each dataset-condition.
- Multiple comparisons: p values are uncorrected and should not be used as a discovery filter. Effect directions and paired confidence intervals are the main evidence.
- Fallacy scan coverage: 11/11.
  - Simpson's paradox: not indicated; results are reported separately by dataset and condition.
  - Ecological fallacy: not applicable; inference remains at run/dataset level.
  - Berkson's paradox: not indicated.
  - Collider bias: no covariate adjustment was performed.
  - Base-rate neglect: not applicable.
  - Regression to the mean: not indicated; seeds were not selected by prior performance.
  - Survivorship bias: none; all 40 new formal runs completed and all five seeds are retained.
  - Look-elsewhere effect: `CAUTION`; multiple metrics and contrasts were examined.
  - Garden of forking paths: reduced by the frozen configuration and complete 2 x 2 grid, but the analysis remains non-preregistered.
  - Correlation versus causation: the factorial interventions support within-protocol component attribution, not universal causal claims across datasets.
  - Reverse causality: not applicable to the randomized component intervention.

## Reproducibility verdict

- Status: `REPRODUCED` for the 40 new configurations.
- Completion: 40/40, with no crash, timeout, NaN, non-finite loss, or CUDA memory failure.
- Each run retains configuration, data hash, seed, history, model state, embeddings, predictions, timing and peak CUDA memory.

