import argparse
import json
import numpy as np
from qgbmc.evaluation.metrics import clustering_metrics

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('embeddings'); a = p.parse_args()
    with np.load(a.embeddings, allow_pickle=False) as f:
        print(json.dumps(clustering_metrics(f['labels'], f['predicted']), indent=2))
