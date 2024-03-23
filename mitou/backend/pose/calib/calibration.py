import numpy as np
import asyncio

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from .calibfunc import async_calib_linear
from .calibfunc import async_joints2orientations, async_joints2projections
from .calibfunc import visible_from_all_cam
from .preprocess import H36M_BONE
from .utils import triangulate_with_conf

def calculate_n(y, K):
    ni = []
    for i in range(len(K)):
        ni.append(y[i] @ np.linalg.inv(K[i]).T)
    n = np.array(ni)
    return n

async def async_calculate_n(executor, y, K):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, calculate_n, y, K)

async def calibrate(kpts2d_h36m, score2d_h36m, kpts3d, score3d, K):
    # ThreadPoolExecutorを使用
    with ThreadPoolExecutor(max_workers=4) as executor:
        mask_CxNxJ = (score2d_h36m > 0) * (score3d == 1)
        mask_vis_NxJ = visible_from_all_cam(mask_CxNxJ)
        vc = await async_joints2orientations(executor, kpts3d, mask_vis_NxJ, H36M_BONE)
        y = await async_joints2projections(executor, kpts2d_h36m, mask_vis_NxJ)
        n = np.ones((y.shape[0], y.shape[1], 3), dtype=np.float64)
        n[:, :, :2] = y
        n = await async_calculate_n(executor, n, K)

    # 重い処理にはProcessPoolExecutorを使用(将来的にはcalibrateもGPU処理にする)
    with ProcessPoolExecutor(max_workers=4) as executor:
        R_est, t_est, kpts3d_est = await async_calib_linear(executor, vc, n)

    kpts3d_est = kpts3d_est.reshape(-1, 17, 3)
    kpts3d_tri = triangulate_with_conf(kpts2d_h36m, score2d_h36m, K, R_est, t_est, (score2d_h36m > 0))

    return R_est, t_est, kpts3d_est, kpts3d_tri
