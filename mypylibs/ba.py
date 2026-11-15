"""Bundle adjustment (step 5): joint least-squares refinement of poses and 3D points."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


def bundle_adjust(K, Rs, ts, X, obs_cam, obs_pt, obs_uv, free_cams, max_nfev=50, huber_px=1.0):
    """Minimise reprojection error over the poses in free_cams and every observed point.

    Rs (N,3,3), ts (N,3), X (T,3) are updated in place. obs_* list each observation
    (camera index, point index, pixel). Cameras not in free_cams are held fixed
    (they anchor the gauge). Returns the mean reprojection error in px before and after.
    """
    cams, c_loc = np.unique(obs_cam, return_inverse=True)
    pts, p_loc = np.unique(obs_pt, return_inverse=True)
    is_free = np.isin(cams, free_cams)
    free_idx = np.cumsum(is_free) - 1
    nf = int(is_free.sum())

    rv0 = Rotation.from_matrix(Rs[cams]).as_rotvec()
    t0 = ts[cams].copy()
    x0 = np.concatenate([np.hstack([rv0[is_free], t0[is_free]]).ravel(), X[pts].ravel()])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def unpack(x):
        rv, t = rv0.copy(), t0.copy()
        cp = x[:6 * nf].reshape(-1, 6)
        rv[is_free], t[is_free] = cp[:, :3], cp[:, 3:]
        return rv, t, x[6 * nf:].reshape(-1, 3)

    def residuals(x):
        rv, t, P = unpack(x)
        R = Rotation.from_rotvec(rv).as_matrix()
        Xc = np.einsum('nij,nj->ni', R[c_loc], P[p_loc]) + t[c_loc]
        z = np.maximum(Xc[:, 2], 1e-6)
        return np.column_stack([fx * Xc[:, 0] / z + cx - obs_uv[:, 0],
                                fy * Xc[:, 1] / z + cy - obs_uv[:, 1]]).ravel()

    # Each residual depends on one camera (6 params, if free) and one point (3 params).
    n_obs = len(obs_cam)
    r = np.arange(n_obs)
    of = is_free[c_loc]
    rows, cols = [], []
    for k in range(2):
        rr = 2 * r + k
        for j in range(6):
            rows.append(rr[of]); cols.append(6 * free_idx[c_loc[of]] + j)
        for j in range(3):
            rows.append(rr); cols.append(6 * nf + 3 * p_loc + j)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    sparsity = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n_obs, len(x0))).tocsr()

    err = lambda x: float(np.mean(np.linalg.norm(residuals(x).reshape(-1, 2), axis=1)))
    before = err(x0)
    sol = least_squares(residuals, x0, jac_sparsity=sparsity, method='trf', loss='huber',
                        f_scale=huber_px, x_scale='jac', max_nfev=max_nfev)
    rv, t, P = unpack(sol.x)
    Rs[cams], ts[cams], X[pts] = Rotation.from_rotvec(rv).as_matrix(), t, P
    return before, err(sol.x)
