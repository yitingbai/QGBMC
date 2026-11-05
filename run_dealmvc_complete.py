"""Run the official DealMVC network on a frozen complete-view NPZ dataset.

The official repository only exposes BBCSport. This adapter preserves its
network, three-stage training logic, default hyperparameters, fixed batch-size
attention coupling, and evaluation rule while replacing only the data entry,
logging, and output persistence. It does not modify the official checkout.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from qgbmc.evaluation.metrics import clustering_metrics


class MultiViewDataset(Dataset):
    def __init__(self, views, labels):
        self.views = [torch.from_numpy(view.astype(np.float32, copy=False))
                      for view in views]
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return [view[index] for view in self.views], self.labels[index], index


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted((key for key in archive.files if key.startswith('view_')),
                      key=lambda key: int(key[5:]))
        if len(keys) < 2 or 'labels' not in archive.files:
            raise ValueError('NPZ requires view_0... and labels')
        raw = [np.asarray(archive[key], dtype=np.float32) for key in keys]
        labels = np.asarray(archive['labels']).reshape(-1).astype(np.int64)
    permutation = np.random.RandomState(1).permutation(len(labels))
    views = []
    for view in raw:
        if len(view) != len(labels) or not np.isfinite(view).all():
            raise ValueError('invalid view')
        scaled = MinMaxScaler().fit_transform(view)
        views.append(scaled[permutation].astype(np.float32, copy=False))
    return views, labels[permutation]


def match(y_true, y_pred, device):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    dimension = max(y_pred.max(), y_true.max()) + 1
    weights = np.zeros((dimension, dimension), dtype=np.int64)
    for index in range(y_pred.size):
        weights[y_pred[index], y_true[index]] += 1
    rows, columns = linear_sum_assignment(weights.max() - weights)
    mapping = {column: row for row, column in zip(rows, columns)}
    remapped = np.array([mapping[value] for value in y_true], dtype=np.int64)
    return torch.from_numpy(remapped).long().to(device)


@torch.no_grad()
def evaluate(model, dataset, view_count, device, batch_size):
    model.eval()
    predictions = []
    labels = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    for xs, y, _ in loader:
        xs = [x.to(device) for x in xs]
        qs, _, _ = model.forward_cluster(xs)
        q = sum(qs) / view_count
        predictions.append(q.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    predicted = np.concatenate(predictions)
    truth = np.concatenate(labels)
    return clustering_metrics(truth, predicted), truth, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--mse-epochs', type=int, default=300)
    parser.add_argument('--con-epochs', type=int, default=100)
    parser.add_argument('--tune-epochs', type=int, default=50)
    parser.add_argument('--feature-dim', type=int, default=512)
    parser.add_argument('--high-feature-dim', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--ffn-size', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--attention-dropout', type=float, default=0.5)
    args = parser.parse_args()

    if args.batch_size != 256:
        raise ValueError(
            'Official DealMVC couples attention hidden_dim to batch_size=256; '
            'changing batch size would change the network architecture.')
    official_root = Path(args.official_root).resolve()
    data_path = Path(args.data).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(official_root))
    from network import FeedForwardNetwork, MultiHeadAttention, Network

    setup_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('official DealMVC network hardcodes CUDA tensors')
    views, labels = load_npz(data_path)
    dataset = MultiViewDataset(views, labels)
    dims = [view.shape[1] for view in views]
    view_count = len(views)
    class_count = len(np.unique(labels))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    if not len(train_loader):
        raise ValueError('dataset smaller than official batch size')

    model = Network(view_count, dims, args.feature_dim,
                    args.high_feature_dim, class_count, device).to(device)
    attention_net = MultiHeadAttention(
        args.batch_size, args.attention_dropout, args.num_heads,
        6).to(device)
    p_net = FeedForwardNetwork(
        view_count, args.ffn_size, args.attention_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mse = nn.MSELoss()
    p_sample = torch.full((view_count,), 1.0 / view_count, device=device)
    initial_weight = torch.full((view_count, 1), 1.0 / view_count,
                                device=device)
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    for epoch in range(1, args.mse_epochs + 1):
        model.train(); total = 0.0
        for xs, _, _ in train_loader:
            xs = [x.to(device) for x in xs]
            optimizer.zero_grad(set_to_none=True)
            _, _, reconstructed, _ = model(xs)
            loss = sum(mse(x, xr) for x, xr in zip(xs, reconstructed))
            loss.backward(); optimizer.step(); total += loss.item()
        row = {'stage': 'pretrain', 'epoch': epoch,
               'loss': total / len(train_loader)}
        history.append(row)
        if epoch == 1 or epoch == args.mse_epochs or epoch % 50 == 0:
            print(json.dumps(row), flush=True)

    for local_epoch in range(1, args.con_epochs + 1):
        epoch = args.mse_epochs + local_epoch
        model.train(); total = 0.0
        adaptive_weight = initial_weight
        for xs, _, _ in train_loader:
            xs = [x.to(device) for x in xs]
            optimizer.zero_grad(set_to_none=True)
            hs, qs, reconstructed, _ = model(xs)
            losses = []
            local_pseudo = []
            for v in range(view_count):
                for w in range(v + 1, view_count):
                    similarity = torch.exp(torch.mm(hs[v], hs[w].t()))
                    probabilities = similarity / similarity.sum(1, keepdim=True)
                    pseudo = torch.mm(qs[v], qs[w].t())
                    pseudo.fill_diagonal_(1)
                    pseudo = pseudo * (pseudo >= args.threshold).float()
                    pseudo = pseudo / pseudo.sum(1, keepdim=True)
                    local_pseudo.append(pseudo)
                    losses.append(
                        -(torch.log(probabilities + 1e-7) * pseudo).sum(1).mean())
                losses.append(mse(xs[v], reconstructed[v]))

            stacked = torch.cat(
                [torch.mean(h, 1).unsqueeze(1) for h in hs], dim=1).t()
            attention = attention_net(stacked, stacked, stacked)
            learned = p_net(p_sample)
            routing = torch.softmax(attention * learned * adaptive_weight, dim=0)
            adaptive_weight = routing * adaptive_weight
            fusion = torch.zeros_like(hs[0])
            for v in range(view_count):
                fusion = fusion + adaptive_weight[v].item() * hs[v]
            similarity = torch.exp(torch.mm(fusion, fusion.t()))
            probabilities = similarity / similarity.sum(1, keepdim=True)
            pseudo = model.label_contrastive_module(fusion)
            pseudo = torch.mm(pseudo, pseudo.t())
            pseudo.fill_diagonal_(1)
            pseudo = pseudo * (pseudo >= args.threshold).float()
            pseudo = pseudo / pseudo.sum(1, keepdim=True)
            losses.append(
                -(torch.log(probabilities + 1e-7) * pseudo).sum(1).mean())
            if view_count == 2:
                losses.append(F.mse_loss(pseudo, local_pseudo[0]))
            else:
                for v in range(view_count):
                    losses.append(F.mse_loss(pseudo, local_pseudo[v]))
            loss = sum(losses)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite DealMVC objective')
            loss.backward(); optimizer.step(); total += loss.item()
        row = {'stage': 'contrastive', 'epoch': epoch,
               'loss': total / len(train_loader)}
        history.append(row)
        if local_epoch == 1 or local_epoch == args.con_epochs or local_epoch % 25 == 0:
            print(json.dumps(row), flush=True)

    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                             num_workers=0)
    model.eval()
    with torch.no_grad():
        xs, _, _ = next(iter(full_loader))
        xs = [x.to(device) for x in xs]
        hs, _, _, _ = model(xs)
    pseudo_labels = []
    for h in hs:
        representation = MinMaxScaler().fit_transform(h.cpu().numpy())
        pseudo_labels.append(KMeans(
            n_clusters=class_count, n_init=100,
            random_state=args.seed).fit_predict(representation))

    for local_epoch in range(1, args.tune_epochs + 1):
        epoch = args.mse_epochs + args.con_epochs + local_epoch
        model.train(); total = 0.0
        xs, _, _ = next(iter(full_loader))
        xs = [x.to(device) for x in xs]
        optimizer.zero_grad(set_to_none=True)
        _, qs, _, _ = model(xs)
        losses = []
        for v in range(view_count):
            predicted = qs[v].detach().argmax(dim=1).cpu().numpy()
            target = match(pseudo_labels[v], predicted, device)
            losses.append(F.cross_entropy(qs[v], target))
        loss = sum(losses)
        loss.backward(); optimizer.step(); total += loss.item()
        row = {'stage': 'fine_tune', 'epoch': epoch, 'loss': total}
        history.append(row)
        if local_epoch == 1 or local_epoch == args.tune_epochs or local_epoch % 10 == 0:
            print(json.dumps(row), flush=True)

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    metrics, truth, predicted = evaluate(
        model, dataset, view_count, device, args.batch_size)
    commit = subprocess.check_output(
        ['git', '-C', str(official_root), 'rev-parse', 'HEAD'],
        text=True).strip()
    manifest = {
        'method': 'DealMVC',
        'status': 'official-network complete-view adapter',
        'official_root': str(official_root),
        'official_commit': commit,
        'dataset': args.dataset_name,
        'data_path': str(data_path),
        'data_sha256': sha256(data_path),
        'samples': len(dataset), 'views': view_count,
        'input_dims': dims, 'clusters': class_count,
        'seed': args.seed,
        'config': vars(args),
        'metrics': metrics,
        'training_seconds': training_seconds,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'torch': torch.__version__,
        'device': str(device),
        'protocol_notes': [
            'Official repository exposes BBCSport only; adapter replaces data entry.',
            'Official attention hidden dimension is coupled to batch_size=256.',
            'Attention and p networks are instantiated but not optimized, matching train.py.',
            'Method-specific MinMax scaling and fixed RandomState(1) permutation retained.',
        ],
        'source': 'computed baseline; not copied from manuscript',
    }
    (output / 'run.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    with (output / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['stage', 'epoch', 'loss'])
        writer.writeheader(); writer.writerows(history)
    np.savez_compressed(output / 'predictions.npz', labels=truth,
                        predicted=predicted)
    torch.save(model.state_dict(), output / 'model.pt')
    print(json.dumps({'status': 'completed', 'output': str(output),
                      'metrics': metrics}, indent=2), flush=True)


if __name__ == '__main__':
    main()
