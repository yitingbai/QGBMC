## Material Passport

- Experiment ID: `cifar10_four_view_reconstructed_v3`
- Type: transparent reconstructed multi-view extension
- Status: `REPRODUCED` for the stated candidate protocol; not the original manuscript construction
- Samples: 60,000; classes: 10; seeds: 0, 1, 2
- Data SHA-256: `d8de162515a961d3793450177756ef58a5518ff85212bf3028e83571bd7d9461`
- Boundary: newly computed Table 8 evidence; does not replace or validate legacy CIFAR-10 values

# CIFAR-10 four-view reconstructed extension

## Construction

The standard CIFAR-10 train and test splits were concatenated in fixed order. Images were converted to RGB, resized to 256, center-cropped to 224, and ImageNet-normalized. Four ImageNet-pretrained feature extractors were used:

| View | Extractor | Weights | Dimension |
|---:|---|---|---:|
| 0 | VGG19 | IMAGENET1K_V1 | 4096 |
| 1 | DenseNet121 | IMAGENET1K_V1 | 1024 |
| 2 | ResNet101 | IMAGENET1K_V2 | 2048 |
| 3 | ResNet50 | IMAGENET1K_V2 | 2048 |

The original CIFAR archive was obtained from the pinned MIT Open Learning mirror commit `a48007227f9e2cd0af96175f4afb5ac4965e261b`. The archive MD5 was `c58f30108f718f92721af3b95e74349a`, matching torchvision's integrity check, and its SHA-256 was `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.

Labels were not used in feature extraction or representation learning. They were used only for post-training clustering evaluation.

## Frozen protocol

- Full and uniform-quality variants
- Clean and element-wise Gaussian replacement of view 0 at rate 0.9
- Seeds 0, 1, 2
- 50 epochs; batch 256; Adam 1e-3; hidden [256,128]; latent 64
- Relative quality, gamma=2, tau=exp(-1), weighted SCA, per-feature reconstruction
- MiniBatchKMeans with three initializations

## Table 8 candidate results

Mean ± sample SD over three seeds.

| Variant | Condition | ACC | NMI | ARI | PUR |
|---|---|---:|---:|---:|---:|
| full | clean | 0.6246 ± 0.0043 | 0.5515 ± 0.0030 | 0.4413 ± 0.0025 | 0.6412 ± 0.0064 |
| uniform quality | clean | 0.5675 ± 0.0240 | 0.5400 ± 0.0232 | 0.4009 ± 0.0356 | 0.5977 ± 0.0116 |
| full | element 0.9 | 0.6687 ± 0.0464 | 0.6048 ± 0.0229 | 0.4911 ± 0.0333 | 0.6817 ± 0.0399 |
| uniform quality | element 0.9 | 0.7029 ± 0.0558 | 0.6183 ± 0.0344 | 0.5102 ± 0.0413 | 0.7116 ± 0.0426 |

## Paired comparisons

- Full minus uniform, clean:
  - ACC +0.0571, 95% CI [-0.0029, +0.1170]
  - NMI +0.0115, 95% CI [-0.0448, +0.0679]
  - ARI +0.0404, 95% CI [-0.0435, +0.1243]
  - PUR +0.0435, 95% CI [+0.0160, +0.0711]
- Full minus uniform, rate 0.9:
  - ACC -0.0342, 95% CI [-0.0882, +0.0198]
  - NMI -0.0135, 95% CI [-0.0689, +0.0420]
  - ARI -0.0191, 95% CI [-0.0669, +0.0288]
  - PUR -0.0300, 95% CI [-0.0630, +0.0031]

With only three seeds and four exploratory metrics, these intervals are descriptive and are not used as confirmatory significance claims.

## Mechanism diagnostics

Three-seed, 50-epoch means for the full variant:

| Condition | View | Quality | Modulation | Decay occupancy |
|---|---:|---:|---:|---:|
| clean | 0 | 0.4868 | 1.4751 | 0.2030 |
| clean | 1 | 0.1358 | 0.1369 | 0.9995 |
| clean | 2 | 0.9971 | 1.0002 | 0.0022 |
| clean | 3 | 0.2985 | 0.7641 | 0.8616 |
| element 0.9 | 0 | 0.1353 | 0.1353 | 1.0000 |
| element 0.9 | 1 | 0.6175 | 1.4865 | 0.0005 |
| element 0.9 | 2 | 0.9999 | 1.0001 | 0.0000 |
| element 0.9 | 3 | 0.7520 | 1.2890 | 0.0001 |

## Evidence-bounded interpretation

1. Under rate-0.9 corruption, the designated view 0 is consistently assigned the minimum relative quality and enters the decay branch in every batch. This verifies the intended routing response.
2. The full variant does not outperform uniform quality under strong corruption. All four mean differences are negative and their n=3 confidence intervals include zero.
3. Strong corruption sometimes improves the aggregate clustering metrics, but the effect is variable and cannot be interpreted as noise being beneficial.
4. Under clean data, view 1 is suppressed in 99.95% of batches and view 3 in 86.16%. The relative quality proxy can therefore suppress an uncorrupted but comparatively weaker view.
5. The correct manuscript conclusion is conditional mechanism activation with dataset-dependent downstream benefit, not universal robustness.
6. This protocol is transparent and reproducible but does not identify how the unavailable original CIFAR-10 four-view experiment was constructed.

## Audit trail

- Builder: `experiment/build_reconstructed_image_views.py`
- Data: `data/cifar4_reconstructed_60000.npz`
- Manifest: `data/cifar4_reconstructed_60000.manifest.json`
- Formal runs: `runs/cifar4_reconstructed__{full,uniform_quality}__{clean,element0.9}__v3__e50__s{0..2}`
- Engineering smoke outputs are excluded from reported results.
