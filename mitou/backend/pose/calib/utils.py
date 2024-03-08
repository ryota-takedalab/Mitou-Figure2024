import numpy as np
import torch
from pycalib.calib import rebase_all
from scipy.spatial.transform import Rotation


def constraint_mat_from_single_view(p, proj_mat):
    u, v = p
    const_mat = np.empty((2, 4))
    const_mat[0, :] = u * proj_mat[2, :] - proj_mat[0, :]
    const_mat[1, :] = v * proj_mat[2, :] - proj_mat[1, :]

    return const_mat[:, :3], -const_mat[:, 3]


def constraint_mat(p_stack, proj_mat_stack):
    lhs_list = []
    rhs_list = []
    for p, proj in zip(p_stack, proj_mat_stack):
        lhs, rhs = constraint_mat_from_single_view(p, proj)
        lhs_list.append(lhs)
        rhs_list.append(rhs)
    A = np.vstack(lhs_list)
    b = np.hstack(rhs_list)
    return A, b


def triangulate_point(p_stack, proj_mat_stack, confs=None):
    A, b = constraint_mat(p_stack, proj_mat_stack)
    if confs is None:
        confs = np.ones(b.shape)
    else:
        confs = np.array(confs).repeat(2)

    p_w, _, rank, _ = np.linalg.lstsq(A * confs.reshape((-1, 1)), b * confs, rcond=None)
    if np.sum(confs > 0) <= 2:
        return np.full((3), np.nan)

    if rank < 3:
        raise Exception("not enough constraint")
    return p_w


def triangulate_with_conf(p2d, s2d, K, R_w2c, t_w2c, mask):
    R_w2c, t_w2c = rebase_all(R_w2c, t_w2c, normalize_scaling=True)

    assert p2d.ndim == 4
    assert s2d.ndim == 3

    Nc, Nf, Nj, _ = p2d.shape
    P_est = []
    for i in range(Nc):
        P_est.append(K[i] @ np.hstack((R_w2c[i], t_w2c[i])))
    P_est = np.array(P_est)

    X = []
    for i in range(Nf):
        for j in range(Nj):
            x = p2d[:, i, j, :].reshape((Nc, 2))
            m = mask[:, i, j]
            confi = s2d[:, i, j]

            if confi.sum() > 0 and m.sum() > 1:
                x3d = triangulate_point(x[m], P_est[m], confi[m])
            else:
                x3d = np.full(4, np.nan)
            X.append(x3d[:3])

    X = np.array(X)
    X = X.reshape(Nf, Nj, 3)
    return X


def invRT(R, t):
    T = np.eye(4)
    if t.shape == (3, 1):
        t = t[:, -1]

    T[:3, :3] = R
    T[:3, 3] = t
    invT = np.linalg.inv(T)
    invR = invT[0:3, 0:3]
    invt = invT[0:3, 3]
    return invR, invt


def invRT_batch(R_w2c, t_w2c):
    t_c2w = []
    R_c2w = []

    if len(t_w2c.shape) == 2:
        t_w2c = t_w2c[:, :, None]

    for R_w2c_i, t_w2c_i in zip(R_w2c, t_w2c):
        R_c2w_i, t_c2w_i = invRT(R_w2c_i, t_w2c_i)
        R_c2w.append(R_c2w_i)
        t_c2w.append(t_c2w_i)

    t_c2w = np.array(t_c2w)
    R_c2w = np.array(R_c2w)

    return R_c2w, t_c2w


def get_c2w_params(R_c2w, t_c2w):
    N = R_c2w.shape[0]
    orientations = []
    translations = []
    for n in range(N):
        quat = Rotation.from_matrix(R_c2w[n, :, :]).as_quat()
        orientation = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
        translation = t_c2w[n, :].astype(np.float32)
        orientations.append(orientation)
        translations.append(translation)

    return orientations, translations


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) is np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) is torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) is torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R)
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)
