"""Inspect and validate aligned multi-view NPZ files bound in data_contract.md."""
import argparse
import hashlib
import json
from pathlib import Path

from qgbmc.data.io import load_npz, preprocess


def describe(path):
    path = Path(path)
    views, labels = load_npz(path)
    return {
        'path': str(path.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'samples': len(views[0]),
        'view_dims': [int(x.shape[1]) for x in views],
        'labels_present': labels is not None,
        'classes': int(len(set(labels.tolist()))) if labels is not None else None,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    print(json.dumps(describe(args.path), indent=2))

