"""Extract representations for clean aligned features using saved normalization."""
import argparse
from pathlib import Path
import numpy as np
import torch
from qgbmc.data.io import load_npz
from qgbmc.models.network import QGBMC
from qgbmc.engine.trainer import embed

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    root = Path(a.run)
    state = torch.load(root / 'model.pt', map_location='cpu', weights_only=True)
    cfg = state['config']
    model = QGBMC(state['input_dims'], cfg['hidden'], cfg['latent'])
    model.load_state_dict(state['model'])
    views, labels = load_npz(a.data)
    if [x.shape[1] for x in views] != state['input_dims']:
        raise ValueError('input feature dimensions differ from checkpoint')
    with np.load(root / 'preprocessing.npz', allow_pickle=False) as f:
        views = [((x-f[f'mean_{i}'])/f[f'std_{i}']).astype('float32') for i,x in enumerate(views)]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, embedding=embed(model, views, torch.device('cpu')),
                        **({'labels':labels} if labels is not None else {}))

if __name__ == '__main__':
    main()
