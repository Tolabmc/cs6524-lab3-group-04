"""Step 2 study: compare SIFT, ORB and KLT tracking on both videos.

Run from the repository root (not part of main.py):
    python -m mypylibs.feature_study

Writes:
    out/step2_features.csv                    one row per (video, mask, method)
    assets/latex/step2_track_survival.png     tracks surviving >= L frames (report figure)
"""

import csv
import time
from os import path as opath

import numpy as np

from mypylibs.features import FeatureParams, build_tracks, detect_and_link, inside, object_mask
from mypylibs.video import load_calibration, load_frames, undistort_frames, undistort_valid_mask

ROOT_IN = 'in'
OUT_CSV = opath.join('out', 'step2_features.csv')
OUT_FIG = opath.join('assets', 'latex', 'step2_track_survival.png')

METHODS = {
    'SIFT': FeatureParams(method='sift', ratio=0.75),
    'ORB': FeatureParams(method='orb', ratio=0.8),
    'KLT': FeatureParams(method='klt'),
}
RUNS = [('a', 'none'), ('a', 'object'), ('b', 'none'), ('b', 'object')]
FIG_RUNS = {('a', 'none'): 'a: camera orbits, full frame', ('b', 'object'): 'b: object spins, object mask'}
# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light mode).
STYLE = {'SIFT': ('#2a78d6', '-'), 'ORB': ('#eb6834', '--'), 'KLT': ('#1baf7a', ':')}


def evaluate(res, obj_masks) -> dict:
    ids, lengths = build_tracks(res)
    verified = np.array([len(l) for l in res.links])
    cand = np.array(res.n_candidates)
    on_obj = [inside(obj_masks[i], res.pts[i][l[:, 0]]).mean() for i, l in enumerate(res.links) if len(l)]
    usable = [np.sum(lengths[f_ids] >= 5) for f_ids in ids]
    return {
        'pts_per_frame': np.mean([len(p) for p in res.pts]),
        'verified_per_pair': verified.mean(),
        'verified_min': verified.min(),
        'weakest_pair': int(verified.argmin()),
        'inlier_ratio': verified.sum() / max(cand.sum(), 1),
        'on_object_share': float(np.mean(on_obj)) if on_obj else 0.0,
        'tracks_ge5': int(np.sum(lengths >= 5)),
        'tracks_ge10': int(np.sum(lengths >= 10)),
        'tracks_ge20': int(np.sum(lengths >= 20)),
        'longest_track': int(lengths.max()),
        'usable_pts_per_frame': float(np.mean(usable)),
    }, lengths


def plot(survival: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 7, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True, constrained_layout=True)
    L = np.arange(2, 61)
    for ax, (run, title) in zip(axes, FIG_RUNS.items()):
        for name, (color, ls) in STYLE.items():
            lengths = survival[(run, name)]
            n = np.array([np.sum(lengths >= l) for l in L])
            alive = n > 0  # stop the line where the last track ends (log axis has no 0)
            ax.plot(L[alive], n[alive], color=color, ls=ls, lw=1.2, label=name)
            if alive[-1]:  # direct label only where the line is still alive at the right edge
                ax.annotate(name, (L[-1], n[-1]), xytext=(2, 0), textcoords='offset points',
                            va='center', fontsize=6, color='#0b0b0b')
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.8)
        ax.set_title(title, fontsize=7, color='#0b0b0b')
        ax.set_xlabel('min. track length $L$ (frames)')
        ax.set_xlim(2, 70)
        ax.grid(True, which='major', color='#e4e3df', lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('tracks with length $\\geq L$')
    axes[1].legend(frameon=False, fontsize=6, loc='upper right')
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)


def main() -> None:
    K, D = load_calibration(ROOT_IN)
    rows, survival = [], {}
    for video in sorted({v for v, _ in RUNS}):
        frames = undistort_frames(load_frames(opath.join(ROOT_IN, f'{video}.mp4')), K, D)
        valid = undistort_valid_mask(K, D, frames[0].shape)
        obj_masks = [object_mask(f) for f in frames]
        for v, mask in RUNS:
            if v != video:
                continue
            for name, base in METHODS.items():
                p = FeatureParams(**{**base.__dict__, 'mask': mask})
                t0 = time.perf_counter()
                res = detect_and_link(frames, K, p, valid)
                secs = time.perf_counter() - t0
                stats, lengths = evaluate(res, obj_masks)
                survival[((video, mask), name)] = lengths
                rows.append({'video': video, 'mask': mask, 'method': name, **stats, 'seconds': secs})
                print(f"{video} {mask:6s} {name:4s} | pts/f {stats['pts_per_frame']:6.0f} | "
                      f"verified/pair {stats['verified_per_pair']:5.0f} (min {stats['verified_min']:3d} "
                      f"@{stats['weakest_pair']:3d}) | inl {stats['inlier_ratio']:.0%} | "
                      f"on-object {stats['on_object_share']:4.0%} | tracks>=5 {stats['tracks_ge5']:5d} "
                      f">=10 {stats['tracks_ge10']:4d} >=20 {stats['tracks_ge20']:4d} max {stats['longest_track']:3d} | "
                      f"usable/f {stats['usable_pts_per_frame']:5.0f} | {secs:4.1f}s", flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plot(survival)


if __name__ == '__main__':
    main()
