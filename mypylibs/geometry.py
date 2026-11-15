"""Two-view geometry (steps 3 and 4).

Pose convention: world -> camera, x_cam = R @ X + t, projection P = K [R | t].
"""

import cv2
import numpy as np


def projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, np.reshape(t, (3, 1))])


def centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera centre in world coordinates."""
    return -R.T @ t


def rotation_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def relative_pose(K: np.ndarray, p1: np.ndarray, p2: np.ndarray, ransac_px: float):
    """Step 3: essential matrix (RANSAC) and the (R, t) it decomposes to.

    Returns (R, t, inliers) with |t| = 1, or None. inliers are matches that fit E and
    triangulate in front of both cameras.
    """
    if len(p1) < 8:
        return None
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_px)
    if E is None or mask is None:
        return None
    E = E[:3]  # with very few points OpenCV may stack several solutions
    _, R, t, mask = cv2.recoverPose(E, p1, p2, K, mask=mask.copy())
    return R, t.ravel(), mask.ravel() > 0


def triangulate(K, R1, t1, R2, t2, uv1, uv2) -> np.ndarray:
    """Linear triangulation of matched pixels seen by two cameras -> (n, 3) world points."""
    if len(uv1) == 0:
        return np.empty((0, 3))
    Xh = cv2.triangulatePoints(projection(K, R1, t1), projection(K, R2, t2),
                               np.asarray(uv1, np.float64).T, np.asarray(uv2, np.float64).T)
    return (Xh[:3] / Xh[3]).T


def reprojection_error(K, R, t, X, uv) -> tuple[np.ndarray, np.ndarray]:
    """Pixel error of each point re-projected into a camera, and its depth in that camera."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    proj = Xc[:, :2] / zs[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return np.linalg.norm(proj - uv, axis=1), z


def parallax_deg(X: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """Angle between the two viewing rays of each point (small angle = unreliable depth)."""
    a, b = X - C1, X - C2
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def triangulate_checked(K, R1, t1, R2, t2, uv1, uv2, max_reproj_px, min_parallax_deg):
    """Triangulate and keep points in front of both cameras, re-projecting well, with enough parallax."""
    X = triangulate(K, R1, t1, R2, t2, uv1, uv2)
    e1, z1 = reprojection_error(K, R1, t1, X, uv1)
    e2, z2 = reprojection_error(K, R2, t2, X, uv2)
    ok = (z1 > 0) & (z2 > 0) & (e1 < max_reproj_px) & (e2 < max_reproj_px)
    ok &= parallax_deg(X, centre(R1, t1), centre(R2, t2)) >= min_parallax_deg
    return X, ok, e1, e2
