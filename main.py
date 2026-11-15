"""CS6524 Lab 3: Structure from Motion.

Runs the full pipeline on both videos for every sensitivity setting (step 7):
focal length x {0.8, 1.0, 1.2}, with and without undistortion -> 12 point clouds
out/{a,b}_{08,10,12}{u,d}.ply, plus out/metrics.csv with one row per run.
"""

import csv
import time
import traceback
from os import path as opath

import numpy as np
import yaml

from mypylibs.dense import DenseParams, dense_cloud
from mypylibs.features import FeatureParams, detect_and_link
from mypylibs.ply import write_ply
from mypylibs.sfm import SfmParams, run_sfm
from mypylibs.video import load_calibration, load_frames, undistort_frames, undistort_valid_mask

ROOT_IN = 'in'
ROOT_OUT = 'out'
VIDEOS = ['a', 'b']
FOCAL_FACTORS = [0.8, 1.0, 1.2]


def run(video, frames_raw, K0, D, factor, undistort, cfg) -> dict:
    K = K0.copy()
    K[0, 0] *= factor
    K[1, 1] *= factor  # the scaled K is used everywhere, including undistortion
    if undistort:
        frames = undistort_frames(frames_raw, K, D)
        valid = undistort_valid_mask(K, D, frames[0].shape)
    else:
        frames, valid = frames_raw, None

    t0 = time.perf_counter()
    res = detect_and_link(frames, K, FeatureParams(**cfg['features']), valid)       # step 2
    sparse = run_sfm(res, K, SfmParams(**cfg['sfm']))                               # steps 3-5
    xyz, rgb, n_raw = dense_cloud(frames, K, sparse['Rs'], sparse['ts'],            # step 6
                                  DenseParams(**cfg['dense']), valid)
    if len(xyz) == 0:  # dense failed: fall back to the coloured sparse cloud
        xyz = sparse['X']
        rgb = np.array([frames[f][int(round(v)), int(round(u))][::-1]
                        for f, (u, v) in zip(sparse['X_frame'], sparse['X_uv'])], np.uint8).reshape(-1, 3)

    name = f"{video}_{round(factor * 10):02d}{'u' if undistort else 'd'}"
    write_ply(opath.join(ROOT_OUT, f'{name}.ply'), xyz, rgb)
    secs = time.perf_counter() - t0
    print(f'    dense: {n_raw} raw -> {len(xyz)} points | wrote {name}.ply | {secs:.1f}s', flush=True)
    return {'run': name, 'video': video, 'focal_factor': factor, 'undistorted': undistort,
            **sparse['stats'], 'dense_raw': n_raw, 'dense_points': len(xyz), 'seconds': round(secs, 1)}


def main():
    K0, D = load_calibration(ROOT_IN)
    rows = []
    for video in VIDEOS:
        with open(opath.join(ROOT_IN, f'{video}.yaml')) as f:
            cfg = yaml.safe_load(f)
        frames = load_frames(opath.join(ROOT_IN, f'{video}.mp4'))
        for undistort in (True, False):
            for factor in FOCAL_FACTORS:
                print(f"[{video}] focal x{factor}, {'undistorted' if undistort else 'distorted'}", flush=True)
                try:
                    rows.append(run(video, frames, K0, D, factor, undistort, cfg))
                except Exception:
                    traceback.print_exc()

    if rows:
        with open(opath.join(ROOT_OUT, 'metrics.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
