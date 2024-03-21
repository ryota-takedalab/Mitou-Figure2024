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
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    await send_progress("Loading model", websocket, client_id)
    yolo_model = 'yolov8x.pt'
    pose_estimator, pose_lifter, dwp_estimator = load_models(device)

    await send_progress("Estimating 2D & 3D pose", websocket, client_id)
    kpts2d, scores2d, kpts3d, scores3d = await inferencer(video_folder, yolo_model, pose_estimator,
                                                    pose_lifter, device)

    await send_progress("Loading camera settings", websocket, client_id)
    # TODO: カメラデバイスに応じたパラメータ変更を可能にする
    param_iphone11 = np.load("./intrinsic/iphone11_4K.npz")
    param_iphone13 = np.load("./intrinsic/iphone13_4K.npz")
    K = np.array([param_iphone11["mtx"], param_iphone13["mtx"], param_iphone13["mtx"]])

    await send_progress("Processing calibration", websocket, client_id)
    R_est, t_est, kpts3d_est, kpts3d_tri = calibrate(kpts2d, scores2d, kpts3d, scores3d, K)
    # Set whole3d to True if you want to visualize the whole body
    whole3d = True
    if whole3d:
        kpts2d_dwp, scores2d_dwp = inferencer_dwp(video_folder, yolo_model, dwp_estimator)
        kpts3d_tri = triangulate_with_conf(kpts2d_dwp, scores2d_dwp, 
                                           K, R_est, t_est, (scores2d_dwp > 0))

    await send_progress("Generating result video", websocket, client_id)
    save_path = "./sample_output/demodemo.mp4"
    ani = vis_calib_res(kpts3d_tri, kpts3d, whole3d)
    ani.save(save_path, writer="ffmpeg")
    print("save mp4 finished")

    return save_path


if __name__ == "__main__":
    main()