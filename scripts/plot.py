import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(); p.add_argument('run'); a = p.parse_args(); root=Path(a.run)
    with (root/'history.csv').open() as f:
        rows=list(csv.DictReader(f))
    x=[int(r['epoch']) for r in rows]
    for title, keys in [('loss',['loss','reconstruction','sca']),
                        ('quality',[k for k in rows[0] if k.startswith('quality_')]),
                        ('modulation',[k for k in rows[0] if k.startswith('modulation_')])]:
        fig, ax=plt.subplots(figsize=(7,4))
        for key in keys: ax.plot(x,[float(r[key]) for r in rows],label=key)
        ax.set(xlabel='Epoch',ylabel=title,title='Computed run: '+title);ax.legend();fig.tight_layout()
        fig.savefig(root/(title+'.png'),dpi=180);plt.close(fig)

if __name__=='__main__':main()
