"""Plot actual repeated experiment summaries, never paper reference values."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('summary'); p.add_argument('--x',required=True)
    p.add_argument('--y',help='second grid parameter for heatmaps')
    a=p.parse_args(); source=Path(a.summary)
    with source.open() as f: rows=list(csv.DictReader(f))
    for metric in ['ACC','NMI','ARI','PUR']:
        if not rows or not all(r[metric+'_mean'] for r in rows):
            continue
        fig,ax=plt.subplots(figsize=(7,4))
        if a.y:
            xs=sorted({float(r[a.x]) for r in rows});ys=sorted({float(r[a.y]) for r in rows})
            grid=np.full((len(ys),len(xs)),np.nan)
            for r in rows:
                i,j=ys.index(float(r[a.y])),xs.index(float(r[a.x]))
                if not np.isnan(grid[i,j]):raise ValueError('duplicate grid cell; select one experiment')
                grid[i,j]=float(r[metric+'_mean'])
            im=ax.imshow(grid,aspect='auto',origin='lower');fig.colorbar(im,ax=ax,label=metric)
            ax.set_xticks(range(len(xs)),xs);ax.set_yticks(range(len(ys)),ys);ax.set_ylabel(a.y)
        else:
            labels=[r[a.x] for r in rows]
            if len(set(labels))!=len(labels):raise ValueError('duplicate x; use --y for a two-parameter grid')
            values=[float(r[metric+'_mean']) for r in rows]
            # No uncertainty bar if fewer than two repeats were available.
            errors=[float(r[metric+'_sd']) if r[metric+'_sd'] else np.nan for r in rows]
            ax.bar(range(len(rows)),values)
            for i,(value,error) in enumerate(zip(values,errors)):
                if np.isfinite(error):ax.errorbar(i,value,yerr=error,color='black',capsize=3)
            ax.set_xticks(range(len(rows)),labels,rotation=35);ax.set_ylabel(metric)
        ax.set_xlabel(a.x);ax.set_title('Computed experiments: '+metric);fig.tight_layout()
        fig.savefig(source.parent/(metric+'_summary.png'),dpi=180);plt.close(fig)

if __name__=='__main__':main()
