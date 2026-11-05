"""Build transparent reconstructed MNIST/CIFAR multi-view feature files.

This script does not claim to recover the unavailable submission-time code.
``mnist3`` follows a published large-scale MVC construction: ImageNet-pretrained
VGG19, DenseNet121 and ResNet101 features. ``cifar4`` is an explicit candidate
protocol using four named ImageNet feature extractors. Labels never enter
feature extraction and are saved only for post-training external evaluation.
"""
import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Must be set before the first CUDA/cuBLAS operation for reproducible GEMMs.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torchvision
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms


MODEL_NAMES = {
    'mnist3': ['vgg19', 'densenet121', 'resnet101'],
    'cifar4': ['vgg19', 'densenet121', 'resnet101', 'resnet50'],
}


def build_dataset(name, root, transform):
    if name == 'mnist3':
        return datasets.MNIST(root=root, train=True, transform=transform,
                              download=True)
    if name == 'cifar4':
        train = datasets.CIFAR10(root=root, train=True, transform=transform,
                                 download=True)
        test = datasets.CIFAR10(root=root, train=False, transform=transform,
                                download=True)
        return ConcatDataset([train, test])
    raise ValueError(name)


def build_model(name):
    if name == 'vgg19':
        weights = models.VGG19_Weights.IMAGENET1K_V1
        model = models.vgg19(weights=weights)
        model.classifier[-1] = nn.Identity()
        dimension = 4096
    elif name == 'densenet121':
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        model.classifier = nn.Identity()
        dimension = 1024
    elif name == 'resnet101':
        weights = models.ResNet101_Weights.IMAGENET1K_V2
        model = models.resnet101(weights=weights)
        model.fc = nn.Identity()
        dimension = 2048
    elif name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        dimension = 2048
    else:
        raise ValueError(name)
    return model.eval(), dimension, weights


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument('--data-root', default='../data/torchvision')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--storage-dtype', choices=['float16', 'float32'],
                        default='float32')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    dataset = build_dataset(args.dataset, Path(args.data_root).resolve(), transform)
    if args.limit is not None:
        if not 1 <= args.limit <= len(dataset):
            raise ValueError('limit outside dataset')
        dataset = Subset(dataset, range(args.limit))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    n_samples = len(dataset)
    storage_dtype = np.dtype(args.storage_dtype)
    labels_reference = None
    view_specs = []
    started_all = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix='reconstructed_views_',
                                     dir=str(output.parent)) as temporary:
        temporary = Path(temporary)
        views = []
        for view_index, model_name in enumerate(MODEL_NAMES[args.dataset]):
            model, dimension, weights = build_model(model_name)
            model = model.to(device)
            array = np.lib.format.open_memmap(
                temporary / f'view_{view_index}.npy', mode='w+',
                dtype=storage_dtype, shape=(n_samples, dimension))
            labels = np.empty(n_samples, dtype=np.int64)
            offset = 0
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.inference_mode():
                for images, batch_labels in loader:
                    images = images.to(device, non_blocking=True)
                    features = model(images)
                    if features.ndim != 2 or features.shape[1] != dimension:
                        raise RuntimeError(
                            f'unexpected {model_name} features: {features.shape}')
                    end = offset + len(images)
                    array[offset:end] = features.cpu().numpy().astype(
                        storage_dtype, copy=False)
                    labels[offset:end] = batch_labels.numpy()
                    offset = end
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if offset != n_samples or not np.isfinite(array).all():
                raise RuntimeError('incomplete or non-finite extracted view')
            if labels_reference is None:
                labels_reference = labels
            elif not np.array_equal(labels_reference, labels):
                raise RuntimeError('sample order changed between views')
            array.flush()
            views.append(array)
            view_specs.append({
                'index': view_index,
                'model': model_name,
                'weights': str(weights),
                'dimension': dimension,
                'seconds': elapsed,
                'peak_cuda_reserved_bytes': (
                    torch.cuda.max_memory_reserved(device)
                    if device.type == 'cuda' else 0),
            })
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(json.dumps(view_specs[-1]), flush=True)

        payload = {f'view_{i}': np.asarray(view) for i, view in enumerate(views)}
        payload['labels'] = labels_reference
        np.savez_compressed(output, **payload)
        del payload
        for view in views:
            view.flush()
            if getattr(view, '_mmap', None) is not None:
                view._mmap.close()
        views.clear()
        del array

    manifest = {
        'status': 'reconstructed_protocol_not_original_code',
        'dataset': args.dataset,
        'samples': n_samples,
        'views': view_specs,
        'storage_dtype': str(storage_dtype),
        'preprocessing': 'RGB; resize 256; center crop 224; ImageNet normalization',
        'label_use': 'saved for post-training evaluation only; not used in extraction',
        'seed': 0,
        'device': str(device),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'platform': platform.platform(),
        'total_seconds': time.perf_counter() - started_all,
        'output': str(output),
        'sha256': sha256(output),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    output.with_suffix('.manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
