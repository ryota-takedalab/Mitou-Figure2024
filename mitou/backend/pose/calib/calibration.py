import numpy as np

from .calibfunc import calib_linear
from .calibfunc import joints2orientations, joints2projections
from .calibfunc import visible_from_all_cam
from .preprocess import H36M_BONE
from .utils import triangulate_with_conf


def calibrate(kpts2d_h36m, score2d_h36m, kpts3d, score3d, K):
    mask_CxNxJ = (score2d_h36m > 0) * (score3d == 1)
    mask_vis_NxJ = visible_from_all_cam(mask_CxNxJ)
    vc = joints2orientations(kpts3d, mask_vis_NxJ, H36M_BONE)
    y = joints2projections(kpts2d_h36m, mask_vis_NxJ)

    n = np.ones((y.shape[0], y.shape[1], 3), dtype=np.float64)
    n[:, :, :2] = y
    ni = []
    for i in range(len(K)):
        ni.append(n[i] @ np.linalg.inv(K[i]).T)
    n = np.array(ni)

    R_est, t_est, kpts3d_est = calib_linear(vc, n)
    kpts3d_est = kpts3d_est.reshape(-1, 17, 3)

    kpts3d_tri = triangulate_with_conf(kpts2d_h36m, score2d_h36m,
                                       K, R_est, t_est, (score2d_h36m > 0))

    return R_est, t_est, kpts3d_est, kpts3d_tri
