import glob

import cv2
import copy
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

from ultralytics import YOLO


def process_one_image(img, detector, pose_estimator, bboxs_pre, ids_pre, id=0):
    # detector: YOLOv8x
    det_result = detector.track(img, persist=True, classes=[0], conf=0.3, verbose=False, fraction=1.0)
    if det_result[0].boxes is None:
        print('No person detected!')
        if bboxs_pre is None:
            return None, None, bboxs_pre, ids_pre, id
        if bboxs_pre is not None:
            bboxs = bboxs_pre
            ids = ids_pre
    else:
        bboxs = det_result[0].boxes.xyxy.cpu().numpy()
        ids = det_result[0].boxes.id.cpu().numpy()

    # initialize id
    if id == 0:
        bbox_h = (bboxs[:, 2] - bboxs[:, 0]) * (bboxs[:, 3] - bboxs[:, 1])
        idx = np.argmax(bbox_h)
        id = ids[idx]

    if id not in ids:
        print('tracking person lost!')
        bboxs = bboxs_pre
        ids = ids_pre
    else:
        bboxs_pre = copy.deepcopy(bboxs)
        ids_pre = copy.deepcopy(ids)

    bboxs = np.array(bboxs[ids == id])
    pose_results = inference_topdown(pose_estimator, img, bboxs)
    data_samples = merge_data_samples(pose_results)

    return data_samples.get('pred_instances', None), bboxs, bboxs_pre, ids_pre, id


def estimate2d(input_video, detector, pose_estimator, wholebody=False):
    kpts2d = []
    score2d = []
    bboxs_list = []
    pred_instances_list = []
    bboxs_pre, ids_pre, id = None, None, 0

    cap = cv2.VideoCapture(input_video)
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if frame_idx == 0:
            img_size = frame.shape
        frame_idx += 1

        if not success:
            break

        pred_instances, bboxs, bboxs_pre, ids_pre, id = process_one_image(frame,
                                                                          detector,
                                                                          pose_estimator,
                                                                          bboxs_pre, ids_pre, id)
        pred_instances_list = split_instances(pred_instances)

        kpt = np.array(pred_instances_list[0]['keypoints'])
        score = np.array(pred_instances_list[0]['keypoint_scores'])
        kpts2d.append(kpt)
        score2d.append(score)
        bboxs_list.append(bboxs)

    kpts2d = np.array(kpts2d)
    score2d = np.array(score2d)

    if not wholebody:
        kpts2d, score2d, _ = h36m_coco_format(kpts2d, score2d)

    return kpts2d, score2d, img_size, bboxs_list


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


def inferencer(video_folder, yolo_model, pose_estimator, pose_lifter, device):

    types = ('*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV', '*.AVI')
    video_files = []
    for t in types:
        video_files += glob.glob(video_folder + '/' + t)
    video_files = natsorted(video_files)

    print(video_files)

    kpts2d, scores2d, kpts3d, scores3d = [], [], [], []
    for input_video in video_files:
        detector = YOLO(yolo_model)
        kpt2d, score2d, img_size, bboxs_list = estimate2d(input_video, detector, pose_estimator)
        print(kpt2d.shape)
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


def inferencer_dwp(video_folder, yolo_model, pose_estimator):
    types = ('*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV', '*.AVI')
    video_files = []
    for t in types:
        video_files += glob.glob(video_folder + '/' + t)
    video_files = natsorted(video_files)

    kpts2d, scores2d = [], []
    for input_video in video_files:
        detector = YOLO(yolo_model)
        kpt2d, score2d, img_size, _ = estimate2d(input_video, detector,
                                              pose_estimator, wholebody=True)
        # only add 23 keypoints (17 keypoints + 6 foot keypoints)
        kpts2d.append(kpt2d[:, :23, :])
        scores2d.append(score2d)

    kpts2d = np.array(kpts2d)
    scores2d = np.array(scores2d)

    return kpts2d, scores2d
