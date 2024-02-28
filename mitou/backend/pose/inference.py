import glob

import cv2
import numpy as np
import torch

from mmdet.apis import inference_detector
from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances
from natsort import natsorted

from MotionAGFormer.preprocess import h36m_coco_format
from MotionAGFormer.preprocess import normalize_screen_coordinates
from MotionAGFormer.preprocess import turn_into_clips


def process_one_image(img, detector, pose_estimator):
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.1)]

    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    bboxes = np.array([bboxes[np.argmax(area)]])

    bboxes = bboxes[nms(bboxes, 0.5), :4]
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    return data_samples.get('pred_instances', None)


def estimate2d(input_video, detector, pose_estimator, wholebody=False):
    kpts2d = []
    score2d = []
    pred_instances_list = []
    cap = cv2.VideoCapture(input_video)
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if frame_idx == 0:
            img_size = frame.shape
        frame_idx += 1

        if not success:
            break

        pred_instances = process_one_image(frame, detector, pose_estimator)
        pred_instances_list = split_instances(pred_instances)

        kpt = np.array(pred_instances_list[0]['keypoints'])
        score = np.array(pred_instances_list[0]['keypoint_scores'])
        kpts2d.append(kpt)
        score2d.append(score)

    kpts2d = np.array(kpts2d)
    score2d = np.array(score2d)
    if not wholebody:
        kpts2d, score2d, _ = h36m_coco_format(kpts2d, score2d)

    return kpts2d, score2d, img_size


def estimate3d(pose_lifter, device, kpts2d, score2d, img_size, n_frames=27):
    pose_lifter.eval()
    kpts2d = kpts2d.reshape(1, *kpts2d.shape)
    score2d = score2d.reshape(1, *score2d.shape)
    keypoints = np.concatenate((kpts2d, score2d[..., None]), axis=-1)
    clips, downsample = turn_into_clips(keypoints, n_frames=n_frames)

    kpts3d, score3d = [], []
    for idx, clip in enumerate(clips):
        input2d = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        input2d = torch.from_numpy(input2d.astype('float32')).to(device)

        output_non_flip = pose_lifter(input2d)
        output_flip = pose_lifter(input2d)
        output = (output_non_flip + output_flip) / 2

        if idx == len(clips) - 1:
            output = output[:, downsample]
        output[:, :, 0, :] = 0
        post_out_all = output[0].cpu().detach().numpy()

        for post_out in post_out_all:
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value
            scores = np.ones((17), dtype='float32')

            kpts3d.append(post_out)
            score3d.append(scores)

    return np.array(kpts3d), np.array(score3d)


def inferencer(video_folder, detector, pose_estimator, pose_lifter, device):
    # video_files = natsorted(glob.glob(video_folder + '/*.mp4'))
    video_files = natsorted(glob.glob(video_folder + '/*.mov'))

    print(video_files)

    kpts2d, scores2d, kpts3d, scores3d = [], [], [], []
    for input_video in video_files:
        kpt2d, score2d, img_size = estimate2d(input_video, detector, pose_estimator)
        print("estimated2d finished")

        # モデルサイズに応じてn_framesを変更
        kpt3d, score3d = estimate3d(pose_lifter, device, kpt2d, score2d, img_size, n_frames=27)
        print("estimated3d finished")

        kpts2d.append(kpt2d)
        scores2d.append(score2d)
        kpts3d.append(kpt3d)
        scores3d.append(score3d)

    kpts2d = np.array(kpts2d)
    scores2d = np.array(scores2d)
    kpts3d = np.array(kpts3d)
    scores3d = np.array(scores3d)

    return kpts2d, scores2d, kpts3d, scores3d


def inferencer_dwp(video_folder, detector, pose_estimator):
    # video_files = natsorted(glob.glob(video_folder + '/*.mp4'))
    video_files = natsorted(glob.glob(video_folder + '/*.mov'))

    kpts2d, scores2d = [], []
    for input_video in video_files:
        kpt2d, score2d, img_size = estimate2d(input_video, detector,
                                              pose_estimator, wholebody=True)
        kpts2d.append(kpt2d)
        scores2d.append(score2d)

    kpts2d = np.array(kpts2d)
    scores2d = np.array(scores2d)

    return kpts2d, scores2d
