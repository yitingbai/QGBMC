import argparse
import json
from qgbmc.engine.trainer import train

def main():
    p = argparse.ArgumentParser(description='Train QGBMC on aligned multi-view NPZ')
    p.add_argument('--config', default='configs/default.json')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    print(json.dumps(train(cfg, args.data, args.output), indent=2))

if __name__ == '__main__':
    main()
