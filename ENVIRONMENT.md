# Verified execution environment

The controlled experiments were executed with Python 3.9.21, PyTorch 2.1.2+cu121, and an NVIDIA RTX 4090 with 24 GB memory on Windows. The compact code archive retains the exact Python package pins in `requirements.txt`.

The measured results depend on fixed seeds, dataset hashes, view construction, corruption masks, and configuration values. Reproductions should verify the dataset hashes in `results/data_contract.md` before comparing numerical outputs.
