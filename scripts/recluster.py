"""Re-evaluate a completed run with fused or concatenated latent features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from qgbmc.data.io import load_npz, preprocess
from qgbmc.evaluation.metrics import clustering_metrics
from qgbmc.models.network import QGBMC


@torch.no_grad()
def representations(model, views, device, mode, batch_size=1024):
    chunks = []
    model.eval()
    for start in range(0, len(views[0]), batch_size):
        batch = [torch.as_tensor(x[start:start + batch_size], device=device) for x in views]
        z, fused, _ = model(batch)
        value = fused if mode == 'fused' else torch.cat(z, dim=1)
        chunks.append(value.cpu().numpy())
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--mode', choices=['fused', 'concat'], required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint = torch.load(run_dir / 'model.pt', map_location='cpu')
    cfg = checkpoint['config']
    raw, labels = load_npz(json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))['data_path'])
    views, _, _, _ = preprocess(raw, cfg['noise_rate'], cfg['noise_views'], cfg['seed'],
                                cfg.get('noise_mode', 'element_gaussian'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QGBMC(checkpoint['input_dims'], cfg['hidden'], cfg['latent']).to(device)
    model.load_state_dict(checkpoint['model'])
    embedding = representations(model, views, device, args.mode)
    if len(embedding) >= 50000:
        clusterer = MiniBatchKMeans(n_clusters=cfg['clusters'], n_init=3,
                                    batch_size=8192, random_state=cfg['seed'])
    else:
        clusterer = KMeans(n_clusters=cfg['clusters'], n_init=20,
                           random_state=cfg['seed'])
    predicted = clusterer.fit_predict(embedding)
    result = {
        'mode': args.mode, 'seed': cfg['seed'], 'samples': len(embedding),
        'embedding_dim': embedding.shape[1],
        'metrics': clustering_metrics(labels, predicted),
        'source_run': str(run_dir),
    }
    output = run_dir / f'recluster_{args.mode}.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

