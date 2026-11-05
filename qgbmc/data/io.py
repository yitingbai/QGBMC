from pathlib import Path
import numpy as np

def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        keys = sorted((k for k in f.files if k.startswith('view_')), key=lambda k: int(k[5:]))
        if keys != ['view_' + str(i) for i in range(len(keys))] or len(keys) < 2:
            raise ValueError('need contiguous view_0, view_1, ...; at least two views')
        views = [np.asarray(f[k], dtype=np.float32) for k in keys]
        labels = np.asarray(f['labels']).reshape(-1) if 'labels' in f else None
    n = len(views[0])
    if n < 3 or any(x.ndim != 2 or len(x) != n or x.shape[1] < 1 or not np.isfinite(x).all() for x in views):
        raise ValueError('views must be finite aligned N x D matrices with N >= 3')
    if labels is not None and len(labels) != n:
        raise ValueError('label count does not match samples')
    return views, labels

def preprocess(views, noise_rate=0, noise_views=(), seed=0,
               noise_mode='element_gaussian'):
    if not 0 <= noise_rate <= 1:
        raise ValueError('noise_rate must be in [0,1]')
    if any(i < 0 or i >= len(views) for i in noise_views):
        raise ValueError('noise view index out of range')
    if noise_mode not in {'element_gaussian', 'sample_gaussian'}:
        raise ValueError('noise_mode must be element_gaussian or sample_gaussian')
    rng = np.random.default_rng(seed)
    result, means, stds, masks = [], [], [], []
    for i, x in enumerate(views):
        mean = x.mean(0); std = x.std(0); std[std < 1e-8] = 1
        x = ((x - mean) / std).astype(np.float32)
        if noise_mode == 'element_gaussian':
            mask = (rng.random(x.shape) < noise_rate if i in noise_views
                    else np.zeros(x.shape, dtype=bool))
            # Explicit contract: replace selected standardized entries with N(0,1).
            x[mask] = rng.normal(size=int(mask.sum()))
        else:
            mask = (rng.random(len(x)) < noise_rate if i in noise_views
                    else np.zeros(len(x), dtype=bool))
            # Explicit contract: replace complete standardized sample rows with N(0,1).
            x[mask] = rng.normal(size=(int(mask.sum()), x.shape[1]))
        result.append(x); means.append(mean); stds.append(std); masks.append(mask)
    return result, means, stds, masks

def save_npz(path, views, labels=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {'view_' + str(i): x for i, x in enumerate(views)}
    if labels is not None:
        payload['labels'] = labels
    np.savez_compressed(path, **payload)
