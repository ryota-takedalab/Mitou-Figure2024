import numpy as np

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

    return h36m_kpts[0], h36m_scores[0], valid_frames


def resample(len_frames, n_frames=243):
    even = np.linspace(0, len_frames, num=n_frames, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=len_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints, n_frames=243):
    clips = []
    len_frames = keypoints.shape[1]
    downsample = np.arange(n_frames)
    if len_frames <= n_frames:
        new_indices = resample(len_frames, n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, len_frames, n_frames):
            keypoints_clip = keypoints[:, start_idx:start_idx+n_frames, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != n_frames:
                new_indices = resample(clip_length, n_frames)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2 or X.shape[-1] == 3
    result = np.copy(X)
    result[..., :2] = X[..., :2] / w * 2 - [1, h / w]
    return result
