"""Steps 3-5: consecutive-pair essential matrices, two-view triangulation, and
incremental N-frame reconstruction with windowed + global bundle adjustment.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from mypylibs.ba import bundle_adjust
from mypylibs.features import FeatureResult, build_tracks
from mypylibs.geometry import centre, reprojection_error, relative_pose, rotation_deg, triangulate, triangulate_checked

I3, Z3 = np.eye(3), np.zeros(3)


@dataclass
class SfmParams:
    ransac_px: float = 1.0        # E-RANSAC threshold
    init_gap: int = 4             # second frame of the initial pair (frame 0 is the origin)
    min_track_len: int = 3        # only triangulate tracks seen in at least this many frames
    min_gap: int = 2              # frames between the two views used to triangulate a track
    min_parallax_deg: float = 1.0
    max_reproj_px: float = 2.0
    pnp_reproj_px: float = 2.0
    min_pnp_inliers: int = 8
    split_px: float = 2.0         # cut a track where it stops agreeing with its 3D point (drift)
    ba_every: int = 5             # windowed BA every this many frames (0 = off)
    ba_window: int = 10           # frames in the window; its first 2 are held fixed
    ba_max_nfev: int = 20
    global_ba_max_nfev: int = 40
    outlier_px: float = 4.0       # drop points with an observation worse than this after global BA


class _Tracks:
    """Tracks plus their 3D points, with O(1) lookup of a track's row in each frame.

    Tracks can be split: when a KLT track drifts (specular highlight, silhouette) its
    later observations get a new id, so they no longer corrupt the original 3D point.
    """

    def __init__(self, res: FeatureResult):
        self.pts = res.pts
        self.ids, lengths = build_tracks(res)
        self.n_frames, self.n = len(res.pts), len(lengths)
        cap = 2 * self.n + 1024
        self.row = np.full((self.n_frames, cap), -1, np.int32)
        self.first = np.full(cap, self.n_frames)
        self.lengths = np.zeros(cap, int)
        self.lengths[:self.n] = lengths
        self.X = np.zeros((cap, 3))
        self.has_X = np.zeros(cap, bool)
        for f, ids in enumerate(self.ids):
            self.row[f, ids] = np.arange(len(ids))
            self.first[ids] = np.minimum(self.first[ids], f)

    def uv(self, f: int, tracks: np.ndarray) -> np.ndarray:
        return self.pts[f][self.row[f, tracks]]

    def _grow(self, extra: int) -> None:
        cap = self.row.shape[1]
        if self.n + extra <= cap:
            return
        add = max(cap, extra)
        self.row = np.hstack([self.row, np.full((self.n_frames, add), -1, np.int32)])
        self.first = np.r_[self.first, np.full(add, self.n_frames)]
        self.lengths = np.r_[self.lengths, np.zeros(add, int)]
        self.X = np.vstack([self.X, np.zeros((add, 3))])
        self.has_X = np.r_[self.has_X, np.zeros(add, bool)]

    def split(self, tracks: np.ndarray, f: int) -> None:
        """Give the observations of `tracks` from frame f onwards new track ids."""
        if len(tracks) == 0:
            return
        self._grow(len(tracks))
        new = np.arange(self.n, self.n + len(tracks))
        self.n += len(tracks)
        self.first[new] = f
        for g in range(f, self.n_frames):
            r = self.row[g, tracks]
            m = r >= 0
            if not m.any():
                break
            self.ids[g][r[m]] = new[m]
            self.row[g, new[m]] = r[m]
            self.row[g, tracks[m]] = -1
            self.lengths[new[m]] += 1
            self.lengths[tracks[m]] -= 1


def consecutive_pairs(res: FeatureResult, K, p: SfmParams) -> list:
    """Steps 3+4 for every consecutive pair: E -> (R, t), triangulate with frame i at the
    origin, and measure the two-view reprojection error."""
    out = []
    for i, link in enumerate(res.links):
        p1, p2 = res.pts[i][link[:, 0]], res.pts[i + 1][link[:, 1]]
        rp = relative_pose(K, p1, p2, p.ransac_px)
        if rp is None:
            out.append(None)
            continue
        R, t, inl = rp
        _, ok, e1, e2 = triangulate_checked(K, I3, Z3, R, t, p1[inl], p2[inl], np.inf, 0.0)
        out.append({'R': R, 't': t, 'inliers': int(inl.sum()), 'rot_deg': rotation_deg(R),
                    'reproj_px': float(np.mean(np.r_[e1[ok], e2[ok]])) if ok.any() else np.nan})
    return out


def run_sfm(res: FeatureResult, K: np.ndarray, p: SfmParams, log=print, debug=False) -> dict:
    n = len(res.pts)
    tr = _Tracks(res)
    pairs = consecutive_pairs(res, K, p)

    Rs, ts = np.tile(I3, (n, 1, 1)), np.zeros((n, 3))
    posed = np.zeros(n, bool)
    posed[0] = True

    def add_points(tracks, f0, f1):
        Xn, ok, _, _ = triangulate_checked(K, Rs[f0], ts[f0], Rs[f1], ts[f1], tr.uv(f0, tracks),
                                           tr.uv(f1, tracks), p.max_reproj_px, p.min_parallax_deg)
        tr.X[tracks[ok]] = Xn[ok]
        tr.has_X[tracks[ok]] = True

    def observations(frames):
        cam, pt, uv = [], [], []
        for f in frames:
            ids = tr.ids[f]
            m = tr.has_X[ids]
            cam.append(np.full(m.sum(), f)); pt.append(ids[m]); uv.append(tr.pts[f][m])
        return np.concatenate(cam), np.concatenate(pt), np.concatenate(uv)

    def ba(frames, free, nfev):
        cam, pt, uv = observations(frames)
        X = tr.X[:tr.n]
        e = bundle_adjust(K, Rs, ts, X, cam, pt, uv, free_cams=free, max_nfev=nfev)
        tr.X[:tr.n] = X
        return e

    # Initial pair: frame 0 is the origin, |t| = 1 fixes the (arbitrary) scale.
    g = min(p.init_gap, n - 1)
    common = tr.ids[0][tr.row[g, tr.ids[0]] >= 0]
    rp = relative_pose(K, tr.uv(0, common), tr.uv(g, common), p.ransac_px)
    if rp is None:
        raise RuntimeError('initial pair failed')
    Rs[g], ts[g], inl = rp
    posed[g] = True
    add_points(common[inl & (tr.lengths[common] >= p.min_track_len)], 0, g)

    n_fallback, n_split = 0, 0
    for i in range(1, n):
        ids = tr.ids[i]
        if not posed[i]:
            known = tr.has_X[ids]
            ok = False
            if known.sum() >= p.min_pnp_inliers:
                obj, img = tr.X[ids[known]], tr.pts[i][known].astype(np.float64)
                found, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj, img, K, None, cv2.Rodrigues(Rs[i - 1])[0], ts[i - 1].reshape(3, 1).copy(),
                    useExtrinsicGuess=True, iterationsCount=200, reprojectionError=p.pnp_reproj_px,
                    confidence=0.999)
                if found and inliers is not None and len(inliers) >= p.min_pnp_inliers:
                    k = inliers.ravel()
                    rvec, tvec = cv2.solvePnPRefineLM(obj[k], img[k], K, None, rvec, tvec)
                    Rs[i], ts[i] = cv2.Rodrigues(rvec)[0], tvec.ravel()
                    ok = True
            if not ok:  # fall back to the consecutive-pair E (step 3)
                n_fallback += 1
                pr = pairs[i - 1]
                Rr, trel = (pr['R'], pr['t']) if pr else (I3, Z3)
                scale = _fallback_scale(tr, K, Rs, ts, i, Rr, trel)
                if debug:
                    log(f'      frame {i}: PnP failed (known 3D points {known.sum()}), E fallback, scale {scale:.3f}')
                Rs[i], ts[i] = Rr @ Rs[i - 1], Rr @ ts[i - 1] + scale * trel
            posed[i] = True

            if ok:  # cut tracks that no longer agree with their 3D point (drift)
                known = tr.has_X[ids]
                err, z = reprojection_error(K, Rs[i], ts[i], tr.X[ids[known]], tr.pts[i][known])
                drift = ids[known][(err > p.split_px) | (z <= 0)]
                tr.split(drift, i)
                n_split += len(drift)

        ids = tr.ids[i]
        cand = ids[~tr.has_X[ids] & (tr.lengths[ids] >= p.min_track_len) & (tr.first[ids] <= i - p.min_gap)]
        for f0 in np.unique(tr.first[cand]):
            add_points(cand[tr.first[cand] == f0], f0, i)

        if p.ba_every and i % p.ba_every == 0 and i >= p.ba_window:
            frames = np.arange(i - p.ba_window + 1, i + 1)
            e0, e1 = ba(frames, frames[2:], p.ba_max_nfev)
            if debug:
                log(f'      frame {i}: window BA {e0:.2f} -> {e1:.2f}px, 3D points seen {tr.has_X[tr.ids[i]].sum()}')

    # Global BA over every frame (frame 0 fixed), drop outliers, refine again.
    frames = np.arange(n)
    err0, _ = ba(frames, frames[1:], p.global_ba_max_nfev)
    cam, pt, uv = observations(frames)
    Xc = np.einsum('nij,nj->ni', Rs[cam], tr.X[pt]) + ts[cam]
    proj = Xc[:, :2] / np.maximum(Xc[:, 2:], 1e-9) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    bad = (np.linalg.norm(proj - uv, axis=1) > p.outlier_px) | (Xc[:, 2] <= 0)
    tr.has_X[np.unique(pt[bad])] = False
    _, err2 = ba(frames, frames[1:], p.global_ba_max_nfev)

    stats = {
        'pair_inliers': float(np.mean([q['inliers'] for q in pairs if q])),
        'pair_reproj_px': float(np.nanmean([q['reproj_px'] for q in pairs if q])),
        'pnp_fallbacks': n_fallback,
        'track_splits': n_split,
        'sparse_points': int(tr.has_X.sum()),
        'ba_err_before_px': err0,
        'ba_err_px': err2,
        **loop_metrics(Rs, ts),
    }
    log(f"    sparse: {stats['sparse_points']} pts | 2-view reproj {stats['pair_reproj_px']:.2f}px | "
        f"BA {err0:.2f} -> {err2:.2f}px | PnP fallbacks {n_fallback} | splits {n_split} | "
        f"swept {stats['swept_deg']:.0f} deg | closure gap {stats['closure_gap']:.2f}")

    # Where each sparse point was first seen, so the caller can colour it.
    tracks = np.flatnonzero(tr.has_X[:tr.n])
    f_first = tr.first[tracks]
    uv_first = np.zeros((len(tracks), 2))
    for f in np.unique(f_first):
        m = f_first == f
        uv_first[m] = tr.uv(f, tracks[m])
    return {'Rs': Rs, 'ts': ts, 'X': tr.X[tracks], 'X_frame': f_first, 'X_uv': uv_first,
            'pairs': pairs, 'stats': stats}


def _fallback_scale(tr: _Tracks, K, Rs, ts, i, Rr, trel) -> float:
    """Length of the unit-norm E translation between frames i-1 and i, in map units.

    Triangulate the tracks seen in both frames that already have 3D points using the
    unit baseline, and compare their depths with the map; else reuse the last step length.
    """
    prev = tr.ids[i - 1]
    both = prev[tr.has_X[prev] & (tr.row[i, prev] >= 0)]
    if len(both) >= 3 and np.linalg.norm(trel) > 0:
        Xp = triangulate(K, I3, Z3, Rr, trel, tr.uv(i - 1, both), tr.uv(i, both))
        d_map = (tr.X[both] @ Rs[i - 1].T + ts[i - 1])[:, 2]
        ok = (Xp[:, 2] > 0) & (d_map > 0)
        if ok.sum() >= 3:
            return float(np.median(d_map[ok] / Xp[ok, 2]))
    if i >= 2:
        return float(np.linalg.norm(centre(Rs[i - 1], ts[i - 1]) - centre(Rs[i - 2], ts[i - 2])))
    return 1.0


def loop_metrics(Rs, ts) -> dict:
    """Both videos cover ~360 deg, so the last camera should come back next to the first.

    swept_deg: rotation accumulated about the dominant axis (ideal ~360).
    closure_gap: distance between first and last camera centres / mean orbit radius (ideal ~0).
    """
    rel = Rotation.from_matrix(np.einsum('nij,nkj->nik', Rs[1:], Rs[:-1])).as_rotvec()
    axis = rel.sum(0) / (np.linalg.norm(rel.sum(0)) + 1e-12)
    swept = float(np.degrees(np.sum(rel @ axis)))
    C = -np.einsum('nji,nj->ni', Rs, ts)
    radius = np.mean(np.linalg.norm(C - C.mean(0), axis=1)) + 1e-12
    return {'swept_deg': abs(swept), 'closure_gap': float(np.linalg.norm(C[-1] - C[0]) / radius)}
