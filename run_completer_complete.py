"""Run the official COMPLETER implementation in the complete-data setting.

This adapter changes only the experimental protocol (missing_rate=0 and the
requested seeds/epochs).  The model, losses, data loader and evaluator are
imported from the pinned upstream checkout supplied with ``--official-root``.
"""
import argparse
import itertools
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def repository_commit(root):
    return subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root', required=True)
    parser.add_argument('--dataset', choices=['Caltech101-20', 'Scene_15'], required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    root = Path(args.official_root).resolve()
    sys.path.insert(0, str(root))
    from configure import get_default_config
    from datasets import load_data
    from model import Completer

    config = get_default_config(args.dataset)
    config['dataset'] = args.dataset
    config['training']['missing_rate'] = 0
    config['training']['epoch'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    config['print_num'] = args.epochs

    logger = logging.getLogger('completer-complete')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    arrays, labels = load_data(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x1 = torch.from_numpy(arrays[0]).float().to(device)
    x2 = torch.from_numpy(arrays[1]).float().to(device)
    mask = torch.ones((len(x1), 2), dtype=torch.long, device=device)

    runs = []
    for seed in args.seeds:
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 3)
            torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        model = Completer(config)
        optimizer = torch.optim.Adam(
            itertools.chain(model.autoencoder1.parameters(), model.autoencoder2.parameters(),
                            model.img2txt.parameters(), model.txt2img.parameters()),
            lr=config['training']['lr'])
        model.to_device(device)
        started = time.perf_counter()
        acc, nmi, ari = model.train(
            config, logger, x1, x2, labels, mask, optimizer, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        row = {
            'seed': seed, 'ACC': float(acc), 'NMI': float(nmi), 'ARI': float(ari),
            'training_seconds': elapsed,
            'peak_cuda_allocated_bytes': (torch.cuda.max_memory_allocated(device)
                                          if device.type == 'cuda' else 0),
            'peak_cuda_reserved_bytes': (torch.cuda.max_memory_reserved(device)
                                         if device.type == 'cuda' else 0),
        }
        runs.append(row)
        print(json.dumps(row), flush=True)

    summary = {}
    for metric in ['ACC', 'NMI', 'ARI', 'training_seconds']:
        values = [run[metric] for run in runs]
        summary[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'n_seeds': len(values),
        }
    result = {
        'method': 'COMPLETER',
        'protocol': 'official code; complete data (missing_rate=0)',
        'dataset': args.dataset,
        'official_root': str(root),
        'official_commit': repository_commit(root),
        'config': config,
        'device': str(device),
        'torch': torch.__version__,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'runs': runs,
        'summary': summary,
        'source': 'measured baseline; not copied from the paper',
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
