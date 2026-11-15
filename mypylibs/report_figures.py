"""Report figures from a finished run of main.py (not part of main.py).

    python -m mypylibs.report_figures [results_dir]     # default: out

Writes assets/latex/step7_sensitivity.png and assets/latex/step6_clouds.png.
"""

import csv
import sys
from os import path as opath

import numpy as np

OUT_DIR = opath.join('assets', 'latex')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
# Dataviz reference palette slots 1-2 (validated); marker + line style carry identity too.
SERIES = {True: ('undistorted', '#2a78d6', 'o', '-'), False: ('distorted', '#eb6834', 's', '--')}


def read_ply(path):
    with open(path, 'rb') as f:
        n = 0
        while True:
            line = f.readline().decode('ascii')
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            if line.startswith('end_header'):
                break
        v = np.frombuffer(f.read(), count=n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                                     ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    return np.column_stack([v['x'], v['y'], v['z']]), np.column_stack([v['r'], v['g'], v['b']]) / 255


def _style(plt):
    plt.rcParams.update({'font.size': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2})


def sensitivity(rows, plt):
    metrics = [('swept_deg', 'swept rotation (deg)', 360), ('closure_gap', 'closure gap / radius', 0),
               ('ba_err_px', 'BA reproj. error (px)', None)]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(3.5, 3.6), sharex=True, sharey='row',
                             constrained_layout=True)
    for c, video in enumerate(['a', 'b']):
        for r, (key, label, ideal) in enumerate(metrics):
            ax = axes[r, c]
            if ideal is not None:
                ax.axhline(ideal, color=INK2, lw=0.8, ls=':')
            for und, (name, color, marker, ls) in SERIES.items():
                sel = sorted((float(x['focal_factor']), float(x[key])) for x in rows
                             if x['video'] == video and (x['undistorted'] == 'True') == und)
                if sel:
                    f, y = zip(*sel)
                    ax.plot(f, y, color=color, marker=marker, ms=4, ls=ls, lw=1.2, label=name)
            ax.grid(True, color=GRID, lw=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            if c == 0:
                ax.set_ylabel(label)
            if r == 0:
                ax.set_title(f'video {video}', fontsize=7, color=INK)
        axes[-1, c].set_xlabel('focal length factor')
        axes[-1, c].set_xticks([0.8, 1.0, 1.2])
    axes[1, 0].legend(frameon=False, fontsize=6, loc='upper right')  # empty corner of the panel
    fig.savefig(opath.join(OUT_DIR, 'step7_sensitivity.png'), dpi=300)
    plt.close(fig)


def clouds(results, plt, names=('a_10u', 'b_10u')):
    fig, axes = plt.subplots(len(names), 2, figsize=(3.5, 1.75 * len(names)), constrained_layout=True)
    for r, name in enumerate(names):
        P, C = read_ply(opath.join(results, f'{name}.ply'))
        Q = P - np.median(P, axis=0)
        rng = np.random.default_rng(0)
        sample = Q[rng.choice(len(Q), min(len(Q), 20000), replace=False)]
        Vt = np.linalg.svd(sample, full_matrices=False)[2]
        Q = Q @ Vt.T  # principal axes: 0 = longest extent
        lim = np.percentile(np.abs(Q), 97, axis=0) * 1.15
        for c, (i, j, title) in enumerate([(0, 1, 'side'), (1, 2, 'top')]):
            ax = axes[r, c]
            ax.scatter(Q[:, i], Q[:, j], c=C, s=0.15, linewidths=0, rasterized=True)
            ax.set_xlim(-lim[i], lim[i]); ax.set_ylim(-lim[j], lim[j])
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{name}.ply, {title} ({len(P)} pts)', fontsize=6.5, color=INK)
    fig.savefig(opath.join(OUT_DIR, 'step6_clouds.png'), dpi=300)
    plt.close(fig)


def main(results='out'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    with open(opath.join(results, 'metrics.csv')) as f:
        rows = list(csv.DictReader(f))
    sensitivity(rows, plt)
    clouds(results, plt)


if __name__ == '__main__':
    main(*sys.argv[1:])
