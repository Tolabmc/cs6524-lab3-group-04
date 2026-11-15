"""Step 6: dense reconstruction from the recovered camera poses.

For keyframe pairs (i, i+gap), dense optical flow gives a match for every pixel in the
region of interest; each match is triangulated with the two poses from step 5 and kept
only if it is consistent. Points from all pairs are averaged on a voxel grid.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from mypylibs.features import object_mask
from mypylibs.geometry import triangulate_checked


@dataclass
class DenseParams:
    gap: int = 4                  # frames between the two views of a pair (baseline)
    step: int = 2                 # first frames of consecutive pairs are this far apart
    roi_margin: int = 12          # dilation of the figurine mask (px) defining the region
    fb_px: float = 1.0            # forward-backward flow consistency
    max_reproj_px: float = 1.5
    min_parallax_deg: float = 2.0
    voxel_frac: float = 0.004     # voxel size as a fraction of the cloud's extent
    min_pairs: int = 2            # keep voxels seen by at least this many pairs


def _flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 7, 1.5, 0)


def dense_cloud(frames, K, Rs, ts, p: DenseParams, valid=None):
    """Returns (xyz, rgb, n_raw): voxel-averaged points, their colours, raw point count."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = grays[0].shape
    pts, cols, pair_ids = [], [], []
    for k, i in enumerate(range(0, len(frames) - p.gap, p.step)):
        j = i + p.gap
        roi = object_mask(frames[i], margin=p.roi_margin)
        if valid is not None:
            roi &= valid
        ys, xs = np.nonzero(roi)
        if len(xs) == 0:
            continue
        fwd, bwd = _flow(grays[i], grays[j]), _flow(grays[j], grays[i])
        uv1 = np.column_stack([xs, ys]).astype(np.float64)
        uv2 = uv1 + fwd[ys, xs]
        x2, y2 = np.round(uv2[:, 0]).astype(int), np.round(uv2[:, 1]).astype(int)
        ok = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)
        back = np.full_like(uv1, np.inf)
        back[ok] = uv2[ok] + bwd[y2[ok], x2[ok]]
        ok &= np.linalg.norm(back - uv1, axis=1) < p.fb_px
        X, good, _, _ = triangulate_checked(K, Rs[i], ts[i], Rs[j], ts[j], uv1[ok], uv2[ok],
                                            p.max_reproj_px, p.min_parallax_deg)
        pts.append(X[good])
        cols.append(frames[i][ys[ok][good], xs[ok][good]][:, ::-1])  # BGR -> RGB
        pair_ids.append(np.full(good.sum(), k))
    if not pts:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8), 0
    P, C, ids = np.concatenate(pts), np.concatenate(cols), np.concatenate(pair_ids)
    xyz, rgb = voxel_average(P, C, ids, p.voxel_frac, p.min_pairs)
    return xyz, rgb, len(P)


def voxel_average(P, C, pair_ids, voxel_frac, min_pairs):
    """Average coordinates and colours per voxel; keep voxels supported by >= min_pairs pairs."""
    if len(P) == 0:
        return P, C
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    ext = hi - lo
    keep = np.all((P >= lo - 0.25 * ext) & (P <= hi + 0.25 * ext), axis=1)
    P, C, pair_ids = P[keep], C[keep].astype(float), pair_ids[keep]
    size = voxel_frac * np.linalg.norm(ext) + 1e-12
    q = np.floor((P - P.min(0)) / size).astype(np.int64)
    m = q.max(0) + 1
    key = q[:, 0] + m[0] * (q[:, 1] + m[1] * q[:, 2])
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    xyz = np.column_stack([np.bincount(inv, P[:, d]) / cnt for d in range(3)])
    rgb = np.column_stack([np.bincount(inv, C[:, d]) / cnt for d in range(3)])
    n_pairs = pair_ids.max() + 1
    vp = np.unique(inv * n_pairs + pair_ids)
    support = np.bincount(vp // n_pairs, minlength=len(cnt))
    sel = support >= min_pairs
    return xyz[sel], np.clip(np.round(rgb[sel]), 0, 255).astype(np.uint8)
