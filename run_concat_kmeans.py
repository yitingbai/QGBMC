"""Auditable standardized-view concatenation plus KMeans baseline."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans

from qgbmc.data.io import load_npz, preprocess
from qgbmc.evaluation.metrics import clustering_metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--clusters', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-init', type=int, default=20)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, labels = load_npz(data_path)
    if labels is None:
        raise ValueError('labels required for external evaluation')
    standardized, _, _, _ = preprocess(
        raw, noise_rate=0.0, noise_views=[], seed=args.seed,
        noise_mode='element_gaussian')
    features = np.concatenate(standardized, axis=1)
    started = time.perf_counter()
    predicted = KMeans(
        n_clusters=args.clusters, n_init=args.n_init,
        random_state=args.seed).fit_predict(features)
    seconds = time.perf_counter() - started
    payload = {
        'method': 'Concat-KMeans',
        'status': 'computed simple baseline',
        'dataset': args.dataset_name,
        'data_path': str(data_path),
        'data_sha256': sha256(data_path),
        'samples': len(labels),
        'input_dims': [view.shape[1] for view in raw],
        'concatenated_dim': features.shape[1],
        'preprocessing': 'per-view z-score using QGBMC preprocess; then concatenate',
        'clusters': args.clusters,
        'seed': args.seed,
        'n_init': args.n_init,
        'metrics': clustering_metrics(labels, predicted),
        'clustering_seconds': seconds,
        'numpy': np.__version__,
        'sklearn': sklearn.__version__,
        'platform': platform.platform(),
        'source': 'computed baseline; not copied from manuscript',
    }
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
