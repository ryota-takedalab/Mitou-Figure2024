import numpy as np
import torch
import torch.nn as nn
import json
import asyncio
from fastapi import WebSocket
from progress import send_progress


from mmdet.apis import init_detector
from mmpose.apis import init_model
from mmpose.utils import adapt_mmdet_pipeline

from ultralytics import YOLO

from calib.calibration import calibrate
from calib.utils import triangulate_with_conf
from inference import inferencer, inferencer_dwp
from MotionAGFormer.model import MotionAGFormer
from vis.calibpose import vis_calib_res

def load_models(device):
    pose_config = './mmpose/mmpose_cfg/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
    pose_ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
    pose_estimator = init_model(pose_config, pose_ckpt, device=device)

    dwp_config = './mmpose/mmpose_cfg/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    dwp_ckpt = './mmpose/mmpose_cfg/checkpoint/dw-ll_ucoco_384.pth'
    dwp_estimator = init_model(dwp_config, dwp_ckpt, device=device)

    model_path = "./MotionAGFormer/checkpoint/motionagformer-xs-h36m.pth.tr"
    if model_path.split('/')[-1].split('-')[1] == 'l':
        pose_lifter = MotionAGFormer(n_layers=26, dim_in=3, dim_feat=128,
                                     num_heads=8, neighbour_num=2)
    elif model_path.split('/')[-1].split('-')[1] == 'b':
        pose_lifter = MotionAGFormer(n_layers=16, dim_in=3, dim_feat=128,
                                     num_heads=8, neighbour_num=2)
    elif model_path.split('/')[-1].split('-')[1] == 'xs':
        pose_lifter = MotionAGFormer(n_layers=12, dim_in=3, dim_feat=64,
                                     num_heads=8, neighbour_num=2, n_frames=27)
    pose_lifter = nn.DataParallel(pose_lifter)
    pre_dict = torch.load(model_path)
    pose_lifter.load_state_dict(pre_dict['model'], strict=True)
    pose_lifter.eval()
    pose_lifter.to(device)

    return pose_estimator, pose_lifter, dwp_estimator


async def main(video_folder, websocket=None, client_id=None):
    await send_progress("Loading model", websocket, client_id)
    # 動画が入ったフォルダを指定してアップロードできるように引数を追加
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = YOLO('yolov8x.pt')
    pose_estimator, pose_lifter, dwp_estimator = load_models(device)
    await send_progress("Inferencer", websocket, client_id)


    kpts2d, scores2d, kpts3d, scores3d = inferencer(video_folder,
                                                    detector, pose_estimator, pose_lifter, device)

    await send_progress("Loading camera settings", websocket, client_id)
    # カメラによって異なるので今後変更が必要

    param_iphone11 = np.load("./intrinsic/iphone11_4K.npz")
    param_iphone13 = np.load("./intrinsic/iphone13_4K.npz")
    # K = np.array([param_c13["mtx"], param_c2["mtx"], param_c13["mtx"]])
    K = np.array([param_iphone11["mtx"], param_iphone13["mtx"], param_iphone13["mtx"]])

    await send_progress("Processing calibration", websocket, client_id)

    R_est, t_est, kpts3d_est, kpts3d_tri = calibrate(kpts2d, scores2d, kpts3d, scores3d, K)

    await send_progress("Generating result video", websocket, client_id)

    #kpts2d_dwp, scores2d_dwp = inferencer_dwp(video_folder, detector, dwp_estimator)

    #whole3d = triangulate_with_conf(kpts2d_dwp, scores2d_dwp,
                                    #K, R_est, t_est, (scores2d_dwp > 0))

    #await send_progress("Generating result video", websocket, client_id)
    ani = vis_calib_res(kpts3d_tri, kpts3d)
    save_path = "./sample_output/demodemo.mp4"
    ani.save(save_path, writer="ffmpeg")
    print("save mp4 finished")

    return save_path


if __name__ == "__main__":
    main()