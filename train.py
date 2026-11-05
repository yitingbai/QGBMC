"""Measured experiment launcher with immutable resolved configurations."""
import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

# Required by CUDA >= 10.2 before the first CUDA context is created.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import yaml

from qgbmc.engine.trainer import train as train_one


def _resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def build_config(document, dataset, variant, seed, noise_mode, noise_rate, epochs=None,
                 quality_mode=None, reconstruction_normalization=None,
                 gamma=None, tau=None, lambda_sca=None, noise_views=None,
                 pretrain_epochs=None, latent=None, hidden=None,
                 learning_rate=None, sca_mode=None,
                 log_gradient_norms=False):
    if dataset not in document['datasets']:
        raise KeyError(f'unknown dataset: {dataset}')
    cfg = dict(document['defaults'])
    spec = document['datasets'][dataset]
    cfg.update(spec.get('overrides', {}))
    cfg['clusters'] = int(spec['clusters'])
    cfg['variant'] = variant
    cfg['seed'] = int(seed)
    cfg['noise_mode'] = noise_mode
    cfg['noise_rate'] = float(noise_rate)
    selected_noise_views = (list(noise_views) if noise_views is not None
                            else list(spec.get('noise_views', [0])))
    cfg['noise_views'] = selected_noise_views if noise_rate > 0 else []
    if epochs is not None:
        cfg['epochs'] = int(epochs)
    if quality_mode is not None:
        cfg['quality_mode'] = quality_mode
    if reconstruction_normalization is not None:
        cfg['reconstruction_normalization'] = reconstruction_normalization
    if gamma is not None:
        cfg['gamma'] = float(gamma)
    if tau is not None:
        cfg['tau'] = float(tau)
    if lambda_sca is not None:
        cfg['lambda_sca'] = float(lambda_sca)
    if pretrain_epochs is not None:
        cfg['pretrain_epochs'] = int(pretrain_epochs)
    if latent is not None:
        cfg['latent'] = int(latent)
    if hidden is not None:
        cfg['hidden'] = [int(value) for value in hidden]
    if learning_rate is not None:
        cfg['learning_rate'] = float(learning_rate)
    if sca_mode is not None:
        cfg['sca_mode'] = sca_mode
    if log_gradient_norms:
        cfg['log_gradient_norms'] = True
    if cfg.get('device') == 'auto':
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        cfg['device'] = cfg.get('device', 'cpu')
    return cfg, spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--variant', default='full')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise-mode', choices=['element_gaussian', 'sample_gaussian'], default='element_gaussian')
    parser.add_argument('--noise-rate', type=float, default=0.0)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--quality-mode', choices=['absolute', 'relative'])
    parser.add_argument('--reconstruction-normalization',
                        choices=['per_sample_sum', 'per_feature_mean'])
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--lambda-sca', type=float)
    parser.add_argument('--noise-views', type=int, nargs='+')
    parser.add_argument('--pretrain-epochs', type=int)
    parser.add_argument('--latent', type=int)
    parser.add_argument('--hidden', type=int, nargs='+')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--sca-mode', choices=['gated_positive', 'weighted'])
    parser.add_argument('--log-gradient-norms', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open(encoding='utf-8') as handle:
        document = yaml.safe_load(handle)
    cfg, spec = build_config(document, args.dataset, args.variant, args.seed,
                             args.noise_mode, args.noise_rate, args.epochs,
                             args.quality_mode, args.reconstruction_normalization,
                             args.gamma, args.tau, args.lambda_sca, args.noise_views,
                             args.pretrain_epochs, args.latent, args.hidden,
                             args.learning_rate, args.sca_mode,
                             args.log_gradient_norms)
    base = config_path.parent
    data_path = _resolve(spec['path'], base)
    if args.output:
        output = _resolve(args.output, base)
    else:
        descriptor = (f"{args.dataset}__{args.variant}__{args.noise_mode}__r{args.noise_rate:g}"
               f"__qm{cfg.get('quality_mode','absolute')}"
               f"__rec{cfg.get('reconstruction_normalization','per_sample_sum')}"
               f"__g{cfg['gamma']:g}__t{cfg['tau']:g}__l{cfg['lambda_sca']:g}"
               f"__nv{'-'.join(map(str,cfg['noise_views'])) if cfg['noise_views'] else 'none'}"
               f"__d{cfg['latent']}__h{'-'.join(map(str,cfg['hidden']))}"
               f"__lr{cfg['learning_rate']:g}__pt{cfg.get('pretrain_epochs',0)}"
               f"__sca{cfg.get('sca_mode','gated_positive')}"
               f"__diag{int(cfg.get('log_gradient_norms',False))}"
               f"__e{cfg['epochs']}__s{args.seed}")
        digest = hashlib.sha1(descriptor.encode('utf-8')).hexdigest()[:10]
        tag = (f"{args.dataset}__{args.variant[:8]}__{args.noise_mode[:4]}"
               f"__r{args.noise_rate:g}__cfg{digest}__s{args.seed}")
        output = _resolve(document['run_root'], base) / tag

    try:
        metrics = train_one(cfg, data_path, output)
        print(json.dumps({'status': 'completed', 'output': str(output), 'metrics': metrics}, indent=2))
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            'status': 'failed', 'error_type': type(exc).__name__,
            'error': str(exc), 'traceback': traceback.format_exc(),
            'config': cfg, 'data_path': str(data_path),
        }
        (output / 'failure.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()

