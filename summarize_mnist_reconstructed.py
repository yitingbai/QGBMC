"""Freeze the reconstructed-MNIST extension experiment into auditable tables."""
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ['ACC', 'NMI', 'ARI', 'PUR']
T_CRIT_DF2_975 = 4.302652729911275
CONDITIONS = {
    ('full', 'clean'): 'mnist3_reconstructed__full__clean__v3__e50__s{}',
    ('uniform_quality', 'clean'):
        'mnist3_reconstructed__uniform_quality__clean__v3__e50__s{}',
    ('full', 'element0.9'):
        'mnist3_reconstructed__full__element0.9__v3__e50__s{}',
    ('uniform_quality', 'element0.9'):
        'mnist3_reconstructed__uniform_quality__element0.9__v3__e50__s{}',
}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def paired_summary(left, right):
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    mean = differences.mean(axis=0)
    sd = differences.std(axis=0, ddof=1)
    half_width = T_CRIT_DF2_975 * sd / np.sqrt(len(differences))
    return {
        'per_seed_difference': differences.tolist(),
        'mean_difference': mean.tolist(),
        'sample_sd': sd.tolist(),
        'ci95': np.column_stack((mean - half_width,
                                 mean + half_width)).tolist(),
    }


def main():
    here = Path(__file__).resolve().parent
    latest = here.parent
    run_root = latest / 'runs'
    data_path = latest / 'data' / 'mnist3_reconstructed_60000.npz'
    data_manifest = json.loads(
        data_path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    records = []
    histories = {}
    arrays = {}
    expected_hash = data_manifest['sha256']

    for (variant, condition), pattern in CONDITIONS.items():
        condition_records = []
        condition_histories = []
        for seed in range(3):
            run_dir = run_root / pattern.format(seed)
            manifest = json.loads(
                (run_dir / 'run.json').read_text(encoding='utf-8'))
            if manifest['data_sha256'] != expected_hash:
                raise RuntimeError(f'data hash mismatch: {run_dir}')
            with (run_dir / 'history.csv').open(encoding='utf-8', newline='') as handle:
                history = list(csv.DictReader(handle))
            if len(history) != 50:
                raise RuntimeError(f'expected 50 epochs: {run_dir}')
            record = {
                'variant': variant,
                'condition': condition,
                'seed': seed,
                **{metric: manifest['metrics'][metric] for metric in METRICS},
                'training_seconds': manifest['training_seconds'],
                'clustering_seconds': manifest['clustering_seconds'],
                'run_dir': str(run_dir),
            }
            records.append(record)
            condition_records.append([record[m] for m in METRICS])
            condition_histories.append(history)
        arrays[(variant, condition)] = np.asarray(condition_records)
        histories[(variant, condition)] = condition_histories

    summaries = {}
    for key, values in arrays.items():
        mean, sd = mean_sd(values)
        summaries['/'.join(key)] = {
            'n': len(values), 'metrics': METRICS,
            'mean': mean.tolist(), 'sample_sd': sd.tolist(),
            'per_seed': values.tolist(),
        }

    diagnostics = {}
    for condition in ['clean', 'element0.9']:
        run_means = []
        for history in histories[('full', condition)]:
            columns = [column for column in history[0]
                       if column.startswith(('quality_', 'modulation_',
                                             'decay_branch_'))]
            run_means.append({
                column: float(np.mean([float(row[column]) for row in history]))
                for column in columns
            })
        diagnostics[condition] = {
            column: float(np.mean([item[column] for item in run_means]))
            for column in run_means[0]
        }

    comparisons = {
        'full_minus_uniform_clean': paired_summary(
            arrays[('full', 'clean')],
            arrays[('uniform_quality', 'clean')]),
        'full_minus_uniform_element0.9': paired_summary(
            arrays[('full', 'element0.9')],
            arrays[('uniform_quality', 'element0.9')]),
        'full_noise_minus_clean': paired_summary(
            arrays[('full', 'element0.9')], arrays[('full', 'clean')]),
    }

    output_json = latest / 'mnist3_reconstructed_results_2026-11-03.json'
    output_csv = latest / 'mnist3_reconstructed_runs_2026-11-03.csv'
    output_md = latest / 'mnist3_reconstructed_report_2026-11-03.md'
    payload = {
        'status': 'computed extension; reconstructed protocol; not original code',
        'data_manifest': data_manifest,
        'records': records,
        'summaries': summaries,
        'diagnostics_full_variant': diagnostics,
        'paired_comparisons': comparisons,
        'metrics_scale': '0-1; sample SD; paired CI uses t(df=2)',
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        '# Reconstructed MNIST three-view extension experiment', '',
        '## Version boundary', '',
        '- This is a newly computed extension experiment. It does not recover '
        'the unavailable submission-time scripts.',
        '- It must not replace or alter manuscript Tables 3–7, Figs. 2–5, or '
        'any untraceable O/M values.',
        '- Suggested placement: new Table 8 plus a short protocol paragraph; '
        'all old reported numbers remain historical manuscript values.', '',
        '## Material passport', '',
        f"- Status: `{data_manifest['status']}`",
        f"- Samples: {data_manifest['samples']}; views: 3; dimensions: "
        f"{[view['dimension'] for view in data_manifest['views']]}",
        '- Construction: MNIST training images in fixed order; RGB conversion; '
        'resize 256; center crop 224; ImageNet normalization; ImageNet-pretrained '
        'VGG19, DenseNet121, and ResNet101 feature extractors.',
        '- Published recipe anchor: '
        'https://openreview.net/references/attachment?id=z7vHla0Foa&name=supplementary_material',
        f"- Data SHA-256: `{expected_hash}`",
        '- Labels were not used for extraction or training; they were used only '
        'for post-training external clustering evaluation.', '',
        '## Frozen protocol', '',
        '- Seeds: 0, 1, 2; 50 epochs; Adam 1e-3; batch 256; hidden [256,128]; '
        'latent 64; MiniBatchKMeans with 3 initializations.',
        '- Full variant: relative quality, gamma=2, tau=exp(-1), weighted SCA, '
        'per-feature reconstruction normalization.',
        '- Uniform-quality ablation: identical protocol except quality is fixed '
        'to one for every view.',
        '- Noise: element-wise Gaussian corruption of view 0 at rate 0.9; '
        'clean and noisy runs use matching seeds.', '',
        '## Table 8 candidate: newly computed results (mean ± sample SD, n=3)', '',
        '| Variant | Condition | ACC | NMI | ARI | PUR |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for variant, condition in CONDITIONS:
        result = summaries[f'{variant}/{condition}']
        values = [f'{m:.4f} ± {s:.4f}'
                  for m, s in zip(result['mean'], result['sample_sd'])]
        lines.append(f"| {variant} | {condition} | " + ' | '.join(values) + ' |')

    lines += ['', '## Paired differences', '']
    for name, result in comparisons.items():
        lines.append(f'- {name}:')
        for metric, mean, interval in zip(
                METRICS, result['mean_difference'], result['ci95']):
            lines.append(
                f'  - {metric}: {mean:+.4f}, 95% CI '
                f'[{interval[0]:+.4f}, {interval[1]:+.4f}]')

    lines += ['', '## Mechanism diagnostics for the full variant', '']
    for condition, values in diagnostics.items():
        lines.append(f'- {condition}:')
        for view in range(3):
            lines.append(
                f"  - view {view}: quality={values[f'quality_{view}']:.4f}, "
                f"modulation={values[f'modulation_{view}']:.4f}, "
                f"decay occupancy={values[f'decay_branch_{view}']:.4f}")

    lines += [
        '', '## Evidence-bounded interpretation', '',
        '- Under 0.9 corruption, the contaminated view 0 is assigned mean '
        'quality and modulation 0.1353 and enters the decay branch in 100% of '
        'batches. This directly verifies the intended gating response.',
        '- Full minus uniform under strong noise is positive on all four mean '
        'metrics. The n=3 confidence intervals include zero for ACC, NMI (by a '
        'small margin), and ARI; only PUR excludes zero. With four exploratory '
        'metrics and three seeds, report this as a small, directionally '
        'consistent effect—not a general statistically established gain.',
        '- The full variant scores higher under strong noise than under clean '
        'on average, with wide paired confidence intervals. A defensible '
        'hypothesis is that suppressing view 0 removes a harmful or redundant '
        'view; this experiment does not prove noise is beneficial.',
        '- Clean runs have substantial seed sensitivity. Do not report a '
        'single seed or claim stable clean-data superiority.',
        '- The reconstruction protocol is source-grounded but is not the '
        'original manuscript implementation. These values are suitable only as '
        'a separately labelled extension experiment.', '',
        '## Audit trail', '',
        f'- Machine-readable summary: `{output_json.name}`',
        f'- Per-run table: `{output_csv.name}`',
        '- Every run contains `run.json`, `history.csv`, embeddings, preprocessing '
        'statistics/noise masks, the trained model, and the same data SHA-256.',
    ]
    output_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(output_md), 'json': str(output_json),
                      'csv': str(output_csv), 'runs': len(records)}, indent=2))


if __name__ == '__main__':
    main()
