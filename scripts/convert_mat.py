"""Convert explicit MATLAB keys. No guessed transpose or label indexing."""
import argparse
import numpy as np
from scipy.io import loadmat
from qgbmc.data.io import save_npz, load_npz

def main():
    p = argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output')
    p.add_argument('--views', nargs='+', required=True, help='matrix keys or one MATLAB cell-array key')
    p.add_argument('--labels'); p.add_argument('--label-index', type=int)
    p.add_argument('--transpose', action='store_true'); args = p.parse_args()
    f = loadmat(args.input); views = []
    for key in args.views:
        a = f[key]
        views.extend(list(a.flat) if a.dtype == object else [a])
    views = [np.asarray(x.T if args.transpose else x, dtype=np.float32) for x in views]
    labels = None
    if args.labels:
        labels = f[args.labels]
        if labels.dtype == object:
            if args.label_index is None:
                raise ValueError('object-cell labels require --label-index')
            labels = labels.flat[args.label_index]
        labels = np.asarray(labels).ravel()
    save_npz(args.output, views, labels)
    load_npz(args.output)

if __name__ == '__main__':
    main()
