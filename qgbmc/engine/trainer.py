import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from qgbmc.data.io import load_npz, preprocess
from qgbmc.models.network import QGBMC
from qgbmc.models.losses import objective
from qgbmc.evaluation.metrics import clustering_metrics

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def batches(n, size, generator):
    if size < 3:
        raise ValueError('batch size must be >= 3')
    order = torch.randperm(n, generator=generator)
    parts = list(order.split(size))
    if len(parts) > 1 and len(parts[-1]) < 3:
        tail = parts.pop()
        parts[-1] = torch.cat([parts[-1], tail])
    return parts

@torch.no_grad()
def embed(model, arrays, device, batch_size=1024):
    model.eval()
    return np.concatenate([model([torch.as_tensor(x[i:i+batch_size], device=device) for x in arrays])[1].cpu().numpy()
                           for i in range(0, len(arrays[0]), batch_size)])

def train(cfg, data_path, output):
    allowed = {'full', 'uniform_quality', 'shuffled_quality',
               'wo_vqe', 'wo_qgm', 'wo_sca'}
    if cfg['variant'] not in allowed:
        raise ValueError('unknown ablation variant')
    if cfg['noise_rate'] > 0 and not cfg['noise_views']:
        raise ValueError('nonzero noise_rate requires explicit noise_views')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use an empty output directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    if cfg['epochs'] < 1 or cfg['clusters'] < 2:
        raise ValueError('epochs >=1 and clusters >=2 required')
    seed_all(cfg['seed'])
    device = torch.device(cfg['device'])
    raw, labels = load_npz(data_path)
    views, means, stds, masks = preprocess(
        raw, cfg['noise_rate'], cfg['noise_views'], cfg['seed'],
        cfg.get('noise_mode', 'element_gaussian'))
    if cfg['clusters'] > len(views[0]):
        raise ValueError('clusters exceed sample count')
    dims = [v.shape[1] for v in views]
    model = QGBMC(dims, cfg['hidden'], cfg['latent']).to(device)
    gen = torch.Generator().manual_seed(cfg['seed'])
    pretrain_history = []
    pretrain_started = time.perf_counter()
    if cfg.get('pretrain_epochs', 0) > 0:
        pretrain_parameters = list(model.encoders.parameters()) + list(model.decoders.parameters())
        pretrain_optimizer = torch.optim.Adam(pretrain_parameters, lr=cfg['learning_rate'])
        for epoch in range(1, int(cfg['pretrain_epochs']) + 1):
            model.train(); total = 0.0; count = 0
            for index in batches(len(views[0]), cfg['batch_size'], gen):
                batch = [torch.as_tensor(x[index.numpy()], device=device) for x in views]
                pretrain_optimizer.zero_grad(set_to_none=True)
                latent = [encoder(x) for encoder, x in zip(model.encoders, batch)]
                reconstructed = [decoder(z) for decoder, z in zip(model.decoders, latent)]
                loss = sum((x - xr).square().mean() for x, xr in zip(batch, reconstructed))
                loss.backward(); pretrain_optimizer.step()
                total += loss.item() * len(index); count += len(index)
            row = {'epoch': epoch, 'pretrain_loss': total / count}
            pretrain_history.append(row)
            log_every = max(1, int(cfg.get('log_every', 10)))
            if epoch == 1 or epoch == cfg['pretrain_epochs'] or epoch % log_every == 0:
                print(f"pretrain_epoch={epoch} loss={row['pretrain_loss']:.6f}", flush=True)
    pretraining_seconds = time.perf_counter() - pretrain_started
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    history = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); totals = {}; count = 0
        for index in batches(len(views[0]), cfg['batch_size'], gen):
            batch = [torch.as_tensor(x[index.numpy()], device=device) for x in views]
            optimizer.zero_grad(set_to_none=True)
            loss, details = objective(model, batch, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite objective')
            loss.backward()
            encoder_grad_rms = []
            for encoder in model.encoders:
                squared = loss.new_zeros(())
                count_parameters = 0
                for parameter in encoder.parameters():
                    if parameter.grad is not None:
                        squared = squared + parameter.grad.detach().square().sum()
                        count_parameters += parameter.numel()
                encoder_grad_rms.append(
                    (squared / max(1, count_parameters)).sqrt().item())
            optimizer.step()
            stats = {'loss': loss.item(), **{k: details[k].item() for k in ['reconstruction', 'sca', 'regularization']}}
            for key in ['quality', 'entropy', 'modulation', 'decay_branch']:
                stats.update({f'{key}_{i}': value.item() for i, value in enumerate(details[key])})
            for key in ['align_grad_raw_rms', 'align_grad_modulated_rms']:
                if key in details:
                    stats.update({f'{key}_{i}': value.item()
                                  for i, value in enumerate(details[key])})
            stats.update({f'encoder_grad_rms_{i}': value
                          for i, value in enumerate(encoder_grad_rms)})
            for k, value in stats.items():
                totals[k] = totals.get(k, 0) + value * len(index)
            count += len(index)
        row = {'epoch': epoch, **{k: v / count for k, v in totals.items()}}
        history.append(row)
        log_every = max(1, int(cfg.get('log_every', 10)))
        if epoch == 1 or epoch == cfg['epochs'] or epoch % log_every == 0:
            print(f"epoch={epoch} loss={row['loss']:.6f}", flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    representation = embed(model, views, device)
    cluster_started = time.perf_counter()
    kmeans_mode = cfg.get('kmeans_mode', 'auto')
    use_minibatch = (kmeans_mode == 'minibatch' or
                     (kmeans_mode == 'auto' and len(representation) >= 50000))
    if use_minibatch:
        clusterer = MiniBatchKMeans(
            n_clusters=cfg['clusters'], n_init=cfg.get('kmeans_n_init', 3),
            batch_size=cfg.get('kmeans_batch_size', 8192),
            random_state=cfg['seed'])
    else:
        clusterer = KMeans(
            n_clusters=cfg['clusters'], n_init=cfg.get('kmeans_n_init', 20),
            random_state=cfg['seed'])
    predicted = clusterer.fit_predict(representation)
    clustering_seconds = time.perf_counter() - cluster_started
    metrics = clustering_metrics(labels, predicted) if labels is not None else {}
    np.savez_compressed(output / 'embeddings.npz', embedding=representation, predicted=predicted,
                        **({'labels': labels} if labels is not None else {}))
    np.savez_compressed(output / 'preprocessing.npz', **{f'mean_{i}': v for i,v in enumerate(means)},
                        **{f'std_{i}': v for i,v in enumerate(stds)}, **{f'noise_mask_{i}':v for i,v in enumerate(masks)})
    torch.save({'model': model.state_dict(), 'input_dims': dims, 'config': cfg}, output / 'model.pt')
    with (output / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    if pretrain_history:
        with (output / 'pretrain_history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(pretrain_history[0])); writer.writeheader(); writer.writerows(pretrain_history)
    peak_allocated = (torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
    peak_reserved = (torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
    manifest = {'config': cfg, 'data_path': str(Path(data_path).resolve()),
                'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                'samples':len(views[0]), 'input_dims':dims, 'torch':torch.__version__, 'numpy':np.__version__,
                'metrics':metrics, 'metrics_scale':'0–1, ARI can be negative',
                'training_seconds': training_seconds,
                'pretraining_seconds': pretraining_seconds,
                'clustering_seconds': clustering_seconds,
                'peak_cuda_allocated_bytes': peak_allocated,
                'peak_cuda_reserved_bytes': peak_reserved,
                'clusterer': type(clusterer).__name__,
                'device': str(device), 'platform': platform.platform(),
                'source':'computed run; not paper-reported results'}
    (output / 'run.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return metrics
