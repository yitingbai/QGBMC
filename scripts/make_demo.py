"""Synthetic smoke-test fixture; never a substitute for the eight benchmarks."""
import argparse
import numpy as np
from qgbmc.data.io import save_npz

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', default='data/demo.npz')
    p.add_argument('--samples', type=int, default=240); args = p.parse_args()
    rng = np.random.default_rng(42); y = np.arange(args.samples) % 4
    centers = rng.normal(size=(4, 6)) * 2
    base = centers[y] + rng.normal(size=(args.samples, 6)) * .3
    views = [(base @ rng.normal(size=(6, dim)) + rng.normal(size=(args.samples, dim)) * .1).astype('float32')
             for dim in (12, 16, 10)]
    save_npz(args.output, views, y)

if __name__ == '__main__':
    main()
