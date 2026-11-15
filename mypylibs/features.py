"""Step 2: keypoint detection, consecutive-frame matching and track building.

Every method returns the same representation, so later steps don't depend on it:
  pts[i]   (N_i, 2) float32 pixel coordinates of the points kept in frame i
  links[i] (M_i, 2) int rows (a, b): pts[i][a] and pts[i+1][b] are the same physical
           point, verified as an inlier of the essential matrix between i and i+1
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FeatureParams:
    method: str = 'sift'      # 'sift' | 'orb' | 'klt'
    n_features: int = 2000    # per-frame cap
    ratio: float = 0.75       # Lowe ratio test (sift / orb)
    ransac_px: float = 1.0    # E-RANSAC inlier threshold in pixels
    mask: str = 'none'        # 'none' | 'object'
    static_px: float = 0.0    # drop matches that moved less than this (static-camera video)
    klt_quality: float = 0.01
    klt_min_dist: int = 7
    klt_fb_px: float = 1.0    # forward-backward tracking consistency


@dataclass
class FeatureResult:
    pts: list = field(default_factory=list)
    links: list = field(default_factory=list)
    n_candidates: list = field(default_factory=list)  # matches per pair before E-RANSAC


# Figurine colour (pale yellow-green body) in OpenCV HSV. Body hue is 30-34 in both
# videos; the plywood cabinet in video a is 22-26, so the lower bound sits between them.
BODY_HSV_LO = (27, 40, 100)
BODY_HSV_HI = (45, 255, 255)
DARK_V_MAX = 70  # dark ears / tail / feet


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def object_mask(bgr: np.ndarray, reach: int = 40, margin: int = 12) -> np.ndarray:
    """Mask of the figurine: largest body-coloured blob plus nearby dark parts, convex hull, dilated."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    body = cv2.inRange(hsv, BODY_HSV_LO, BODY_HSV_HI)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body)
    if n < 2:
        return np.zeros(bgr.shape[:2], np.uint8)
    blob = np.uint8(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, DARK_V_MAX)) & cv2.dilate(blob, _disk(reach))
    mask = np.zeros_like(blob)
    cv2.fillConvexPoly(mask, cv2.convexHull(cv2.findNonZero(blob | dark)), 255)
    return cv2.dilate(mask, _disk(margin))


def inside(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Boolean per point: inside the image and on a non-zero mask pixel."""
    h, w = mask.shape
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    ok[ok] = mask[y[ok], x[ok]] > 0
    return ok


def essential_inliers(p1: np.ndarray, p2: np.ndarray, K: np.ndarray, ransac_px: float) -> np.ndarray:
    """Boolean inlier mask of the essential matrix fitted with RANSAC (all False if too few points)."""
    if len(p1) < 8:
        return np.zeros(len(p1), bool)
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_px)
    if E is None or mask is None:
        return np.zeros(len(p1), bool)
    return mask.ravel() > 0


def _verify(p1, p2, K, p: FeatureParams) -> np.ndarray:
    """Indices of matches that move enough (if static_px > 0) and are E-RANSAC inliers."""
    idx = np.arange(len(p1))
    if p.static_px > 0:
        idx = idx[np.linalg.norm(p2 - p1, axis=1) >= p.static_px]
    return idx[essential_inliers(p1[idx], p2[idx], K, p.ransac_px)]


def _descriptor_links(grays, masks, K, p: FeatureParams) -> FeatureResult:
    if p.method == 'sift':
        det, norm = cv2.SIFT_create(nfeatures=p.n_features), cv2.NORM_L2
    elif p.method == 'orb':
        det, norm = cv2.ORB_create(nfeatures=p.n_features), cv2.NORM_HAMMING
    else:
        raise ValueError(p.method)
    matcher = cv2.BFMatcher(norm)

    res = FeatureResult()
    descs = []
    for g, m in zip(grays, masks):
        kps, d = det.detectAndCompute(g, m)
        res.pts.append(np.float32([k.pt for k in kps]).reshape(-1, 2))
        descs.append(d)

    for i in range(len(grays) - 1):
        d1, d2 = descs[i], descs[i + 1]
        pairs = np.empty((0, 2), int)
        if d1 is not None and d2 is not None and len(d1) >= 2 and len(d2) >= 2:
            good = [m[0] for m in matcher.knnMatch(d1, d2, k=2)
                    if len(m) == 2 and m[0].distance < p.ratio * m[1].distance]
            best = {}  # one-to-one: keep the closest match per point of frame i+1
            for m in good:
                if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                    best[m.trainIdx] = m
            pairs = np.array([(m.queryIdx, m.trainIdx) for m in best.values()], int).reshape(-1, 2)
        res.n_candidates.append(len(pairs))
        keep = _verify(res.pts[i][pairs[:, 0]], res.pts[i + 1][pairs[:, 1]], K, p)
        res.links.append(pairs[keep])
    return res


def _new_corners(gray, mask, existing, p: FeatureParams) -> np.ndarray:
    """Shi-Tomasi corners inside mask, away from already-tracked points."""
    m = mask.copy()
    for x, y in np.round(existing).astype(int):
        cv2.circle(m, (int(x), int(y)), p.klt_min_dist, 0, -1)
    budget = p.n_features - len(existing)
    if budget <= 0:
        return np.empty((0, 2), np.float32)
    c = cv2.goodFeaturesToTrack(gray, budget, p.klt_quality, p.klt_min_dist, mask=m)
    return np.empty((0, 2), np.float32) if c is None else c.reshape(-1, 2).astype(np.float32)


def _klt_links(grays, masks, K, p: FeatureParams) -> FeatureResult:
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    res = FeatureResult()
    res.pts.append(_new_corners(grays[0], masks[0], np.empty((0, 2)), p))
    for i in range(len(grays) - 1):
        p0 = res.pts[i]
        a = np.empty(0, int)
        p1 = np.empty((0, 2), np.float32)
        if len(p0):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[i], grays[i + 1], p0, None, **lk)
            pb, stb, _ = cv2.calcOpticalFlowPyrLK(grays[i + 1], grays[i], p1, None, **lk)
            p1 = p1.reshape(-1, 2)
            ok = (st.ravel() == 1) & (stb.ravel() == 1)
            ok &= np.linalg.norm(p0 - pb.reshape(-1, 2), axis=1) < p.klt_fb_px
            ok &= inside(masks[i + 1], p1)
            a = np.flatnonzero(ok)
        res.n_candidates.append(len(a))
        a = a[_verify(p0[a], p1[a], K, p)]
        tracked = p1[a]
        res.pts.append(np.vstack([tracked, _new_corners(grays[i + 1], masks[i + 1], tracked, p)]))
        res.links.append(np.column_stack([a, np.arange(len(a))]))
    return res


def detect_and_link(frames: list[np.ndarray], K: np.ndarray, p: FeatureParams,
                    valid: np.ndarray | None = None) -> FeatureResult:
    """Run step 2 on (already undistorted) BGR frames.

    valid: optional mask of usable pixels (e.g. undistort_valid_mask), ANDed into every frame mask.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    full = np.full(grays[0].shape, 255, np.uint8) if valid is None else valid
    if p.mask == 'object':
        masks = [object_mask(f) & full for f in frames]
    elif p.mask == 'none':
        masks = [full] * len(frames)
    else:
        raise ValueError(p.mask)
    if p.method == 'klt':
        return _klt_links(grays, masks, K, p)
    return _descriptor_links(grays, masks, K, p)


def build_tracks(res: FeatureResult) -> tuple[list[np.ndarray], np.ndarray]:
    """Chain links into tracks.

    Returns ids[i] (track id of every point in frame i) and lengths[t] (frames in track t).
    Unlinked points are tracks of length 1.
    """
    ids = [np.arange(len(res.pts[0]))]
    next_id = len(res.pts[0])
    for i, link in enumerate(res.links):
        cur = np.full(len(res.pts[i + 1]), -1)
        cur[link[:, 1]] = ids[i][link[:, 0]]
        new = cur < 0
        cur[new] = np.arange(next_id, next_id + new.sum())
        next_id += new.sum()
        ids.append(cur)
    lengths = np.bincount(np.concatenate(ids), minlength=next_id)
    return ids, lengths
