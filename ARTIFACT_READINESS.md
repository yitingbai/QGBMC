# Artifact readiness

## Included and checked

- QGBMC source code, baseline wrappers, configurations, aggregation scripts, and table/figure generation utilities are included.
- The environment is pinned in `requirements.txt` and described in `ENVIRONMENT.md`.
- Public-data construction, checksums, view definitions, and corruption protocols are recorded in `results/data_contract.md`.
- `run_inventory.csv` contains 420 sanitized run records.
- All JSON files parse successfully after removal of machine-specific absolute paths.
- The retained test suite passes 29/29 tests with Python 3.9.21.

## Archive boundary

The compact archive excludes public benchmark data and approximately 3.6 GB of raw checkpoints, embeddings, predictions, and corruption masks. The measured summaries and sanitized run inventory are included. The raw-run collection can be deposited as a separate linked record if required by the repository or editor.

## Required before DOI release

1. Select and add an explicit software license.
2. Deposit the archive in a DOI-assigning repository and insert the assigned DOI in the manuscript and response letter.
