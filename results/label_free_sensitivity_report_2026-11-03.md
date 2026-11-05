## Material Passport

- Experiment ID: `v3_label_free_sensitivity_handwritten2_hdigit`
- Type: bounded hyperparameter sensitivity
- Status: `REPRODUCED` for the recorded grid
- Datasets: HandWritten2 and Hdigit
- Seeds: 0, 1, 2
- Total unique configurations: 30, including six reused frozen-base runs and 24 newly computed runs
- Selection boundary: results were not used to change the frozen primary configuration

# Label-free parameter sensitivity

## Protocol

The primary configuration remains gamma=2, tau=exp(-1), and lambda_SCA=1. Two bounded families were evaluated:

1. gamma in {1, 2, 3}, with tau analytically linked as exp(-gamma/2);
2. lambda_SCA in {0.5, 1, 2}, with gamma=2 and tau=exp(-1).

All other architecture, optimization, preprocessing, epoch, and clustering settings were held fixed. Labels were used only for post-training external evaluation and not to select or update the primary configuration.

## Results

Mean ± sample SD over three seeds.

| Dataset | Setting | ACC | NMI | ARI | PUR |
|---|---|---:|---:|---:|---:|
| HandWritten2 | gamma=1 | 0.8205 ± 0.0138 | 0.7897 ± 0.0223 | 0.7151 ± 0.0269 | 0.8205 ± 0.0138 |
| HandWritten2 | gamma=2, lambda=1 (frozen) | 0.8007 ± 0.0015 | 0.7598 ± 0.0092 | 0.6803 ± 0.0069 | 0.8007 ± 0.0015 |
| HandWritten2 | gamma=3 | 0.7680 ± 0.0135 | 0.7181 ± 0.0175 | 0.6259 ± 0.0235 | 0.7680 ± 0.0135 |
| HandWritten2 | lambda=0.5 | 0.7998 ± 0.0073 | 0.7596 ± 0.0126 | 0.6789 ± 0.0154 | 0.7998 ± 0.0073 |
| HandWritten2 | lambda=2 | 0.8245 ± 0.0043 | 0.7979 ± 0.0085 | 0.7237 ± 0.0117 | 0.8245 ± 0.0043 |
| Hdigit | gamma=1 | 0.9936 ± 0.0011 | 0.9814 ± 0.0026 | 0.9859 ± 0.0024 | 0.9936 ± 0.0011 |
| Hdigit | gamma=2, lambda=1 (frozen) | 0.8830 ± 0.0909 | 0.9011 ± 0.0345 | 0.8520 ± 0.0891 | 0.8844 ± 0.0888 |
| Hdigit | gamma=3 | 0.7317 ± 0.0415 | 0.7833 ± 0.0211 | 0.6710 ± 0.0529 | 0.7374 ± 0.0431 |
| Hdigit | lambda=0.5 | 0.7793 ± 0.0114 | 0.8546 ± 0.0048 | 0.7544 ± 0.0054 | 0.7861 ± 0.0046 |
| Hdigit | lambda=2 | 0.9209 ± 0.0489 | 0.9218 ± 0.0179 | 0.8876 ± 0.0538 | 0.9209 ± 0.0489 |

## Evidence-bounded interpretation

1. The method is not parameter-insensitive. Both gamma and lambda materially affect clustering, especially on Hdigit.
2. The frozen gamma=2, lambda=1 configuration is not the post-hoc best setting on either dataset. This supports the integrity of the fixed-protocol comparison but not an optimality claim.
3. The analytic link tau=exp(-gamma/2) removes one independent tuning dimension but does not remove sensitivity to gamma itself.
4. The results support reporting the label-free fixed rule together with its measured sensitivity; they do not support a broad claim of hyperparameter insensitivity.
5. These results belong in Supplementary Information. The main text needs only the fixed rule and a concise statement that performance is parameter-sensitive.

## Audit trail

- Frozen base runs: the formal v3 HandWritten2/Hdigit full-clean runs for seeds 0--2.
- Added runs: `runs/sensitivity__{handwritten2,hdigit}__{g1,g3,l05,l2}__v3__e50__s{0..2}`.
- Every run includes its resolved configuration, data hash, model, history, embeddings, predictions, timing, and memory fields.
