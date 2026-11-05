"""Aggregate completed measured runs without filling missing cells."""
import argparse
import json
import locale
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ('ACC', 'NMI', 'ARI', 'PUR')


def read_run(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except UnicodeDecodeError:
        # Earlier Windows runs used the active system code page. Preserve and
        # read them; all new manifests are explicitly UTF-8.
        return json.loads(path.read_text(encoding=locale.getpreferredencoding(False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='../runs')
    parser.add_argument('--output', default='../results.json')
    args = parser.parse_args()
    root = Path(args.runs).resolve()
    grouped = defaultdict(lambda: defaultdict(list))
    raw_runs = []
    hardware = set()
    for path in sorted(root.glob('*/run.json')):
        if path.parent.name.startswith('smoke_'):
            continue
        run = read_run(path)
        cfg = run['config']
        dataset = path.parent.name.split('__', 1)[0]
        key = '|'.join([
            dataset, cfg['variant'], cfg.get('noise_mode', 'element_gaussian'),
            f"r={cfg.get('noise_rate', 0):g}",
            f"quality={cfg.get('quality_mode', 'absolute')}",
            f"reconstruction={cfg.get('reconstruction_normalization', 'per_sample_sum')}",
            f"gamma={cfg.get('gamma', 3):g}", f"tau={cfg.get('tau', .6):g}",
            f"lambda={cfg.get('lambda_sca', 1):g}",
            f"noise_views={','.join(map(str, cfg.get('noise_views', []))) or 'none'}",
            f"epochs={cfg.get('epochs')}",
            f"latent={cfg.get('latent')}",
            f"hidden={','.join(map(str, cfg.get('hidden', [])))}",
            f"lr={cfg.get('learning_rate')}",
            f"pretrain={cfg.get('pretrain_epochs', 0)}",
            f"sca={cfg.get('sca_mode', 'gated_positive')}",
        ])
        for metric in METRICS:
            if metric in run['metrics']:
                grouped[key][metric].append(float(run['metrics'][metric]))
        hardware.add(run.get('device', 'unknown'))
        raw_runs.append({'path': str(path), 'key': key, 'seed': cfg['seed'],
                         'metrics': run['metrics'],
                         'training_seconds': run.get('training_seconds'),
                         'clustering_seconds': run.get('clustering_seconds'),
                         'peak_cuda_reserved_bytes': run.get('peak_cuda_reserved_bytes')})
    summary = {}
    for key, by_metric in grouped.items():
        summary[key] = {}
        for metric, values in by_metric.items():
            summary[key][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                'n_seeds': len(values),
            }
    result = {
        'task': 'complete multi-view clustering with controlled corruption',
        'dataset': sorted({key.split('|')[0] for key in summary}),
        'metrics': list(METRICS),
        'seeds': sorted({run['seed'] for run in raw_runs}),
        'provenance': {
            'mode': 'measured',
            'source': 'measured runs from experiment/train.py using data_contract.md',
            'dataset_contract': './data_contract.md',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'hardware': sorted(hardware),
        },
        'summary': summary,
        'runs': raw_runs,
        'notes': 'Missing or failed runs are not imputed. Labels were used only after training for external metrics.',
    }
    output = Path(args.output).resolve()
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'output': str(output), 'completed_runs': len(raw_runs)}, indent=2))


if __name__ == '__main__':
    main()

