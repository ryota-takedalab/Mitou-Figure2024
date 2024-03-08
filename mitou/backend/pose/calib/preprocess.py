import numpy as np

H36M_KEY = {
    "Pelvis": 0,
    "RHip": 1,
    "RKnee": 2,
    "RAnkle": 3,
    "LHip": 4,
    "LKnee": 5,
    "LAnkle": 6,
    "Spine": 7,
    "Thorax": 8,
    "Nose": 9,
    "Head": 10,
    "LShoulder": 11,
    "LElbow": 12,
    "LWrist": 13,
    "RShoulder": 14,
    "RElbow": 15,
    "RWrist": 16,
}

H36M_BONE = np.array(
    [
        [H36M_KEY["Head"], H36M_KEY["Nose"]],
        [H36M_KEY["Nose"], H36M_KEY["Thorax"]],
        [H36M_KEY["Thorax"], H36M_KEY["Spine"]],
        [H36M_KEY["Thorax"], H36M_KEY["RShoulder"]],
        [H36M_KEY["Thorax"], H36M_KEY["LShoulder"]],
        [H36M_KEY["RShoulder"], H36M_KEY["RElbow"]],
        [H36M_KEY["LShoulder"], H36M_KEY["LElbow"]],
        [H36M_KEY["RWrist"], H36M_KEY["RElbow"]],
        [H36M_KEY["LWrist"], H36M_KEY["LElbow"]],
        [H36M_KEY["Spine"], H36M_KEY["Pelvis"]],
        [H36M_KEY["RHip"], H36M_KEY["Pelvis"]],
        [H36M_KEY["LHip"], H36M_KEY["Pelvis"]],
        [H36M_KEY["RHip"], H36M_KEY["RKnee"]],
        [H36M_KEY["RKnee"], H36M_KEY["RAnkle"]],
        [H36M_KEY["LHip"], H36M_KEY["LKnee"]],
        [H36M_KEY["LKnee"], H36M_KEY["LAnkle"]],
    ],
    dtype=np.int64,
)

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


def coco_h36m(kpts):
    temporal = kpts.shape[0]
    kpts_h36m = np.zeros_like(kpts, dtype=np.float32)
    htps_kpts = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_kpts: head, thorax, pelvis, spine
    htps_kpts[:, 0, 0] = np.mean(kpts[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_kpts[:, 0, 1] = np.sum(kpts[:, 1:3, 1], axis=1, dtype=np.float32) - kpts[:, 0, 1]
    htps_kpts[:, 1, :] = np.mean(kpts[:, 5:7, :], axis=1, dtype=np.float32)
    htps_kpts[:, 1, :] += (kpts[:, 0, :] - htps_kpts[:, 1, :]) / 3

    htps_kpts[:, 2, :] = np.mean(kpts[:, 11:13, :], axis=1, dtype=np.float32)
    htps_kpts[:, 3, :] = np.mean(kpts[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    kpts_h36m[:, spple_keypoints, :] = htps_kpts
    kpts_h36m[:, h36m_coco_order, :] = kpts[:, coco_order, :]

    kpts_h36m[:, 9, :] -= (kpts_h36m[:, 9, :]
                           - np.mean(kpts[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    kpts_h36m[:, 7, 0] += 2 * (kpts_h36m[:, 7, 0]
                               - np.mean(kpts_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    kpts_h36m[:, 8, 1] -= (np.mean(kpts[:, 1:3, 1], axis=1, dtype=np.float32) - kpts[:, 0, 1])*2/3

    valid_frames = np.where(np.sum(kpts_h36m.reshape(-1, 34), axis=1) != 0)[0]

    return kpts_h36m, valid_frames


def h36m_coco_format(keypoints, scores):
    h36m_kpts = []
    h36m_scores = []
    valid_frames = []
    new_score = np.zeros_like(scores, dtype=np.float32)

    if np.sum(keypoints) != 0.:
        kpts, valid_frame = coco_h36m(keypoints)
        h36m_kpts.append(kpts)
        valid_frames.append(valid_frame)

        new_score[:, h36m_coco_order] = scores[:, coco_order]
        new_score[:, 0] = np.mean(scores[:, [11, 12]], axis=1, dtype=np.float32)
        new_score[:, 8] = np.mean(scores[:, [5, 6]], axis=1, dtype=np.float32)
        new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
        new_score[:, 10] = np.mean(scores[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

        h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)

    return h36m_kpts, h36m_scores, valid_frames
