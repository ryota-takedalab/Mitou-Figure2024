import numpy as np
import torch
import torch.nn as nn
import json
import asyncio
import os
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


class VideoProcessor:
    def __init__(self, video_folder, websocket, client_id):
        self.video_folder = video_folder
        self.websocket = websocket
        self.client_id = client_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = 'yolov8x.pt'
        self.pose_estimator = None
        self.pose_lifter = None
        self.dwp_estimator = None
        self.K = None
        self.R_est = None
        self.t_est = None
        self.kpts3d_tri = None
        self.kpts3d = None

    def load_models(self):
        pose_config = './mmpose/mmpose_cfg/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
        pose_ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
        self.pose_estimator = init_model(pose_config, pose_ckpt, device=self.device)

        dwp_config = './mmpose/mmpose_cfg/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
        dwp_ckpt = './mmpose/mmpose_cfg/checkpoint/dw-ll_ucoco_384.pth'
        self.dwp_estimator = init_model(dwp_config, dwp_ckpt, device=self.device)

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
        pose_lifter.to(self.device)
        self.pose_lifter = pose_lifter
    
    async def init_async(self):
        await self.send_progress("Loading model")
        self.load_models()
    
    async def send_progress(self, message):
        if self.websocket and self.client_id:
            print(f"Sending status '{message}' to client {self.client_id}")
            await self.websocket.send_json({"client_id": self.client_id, "status": message})
            await asyncio.sleep(0)  # イベントループにコントロールを戻す


    async def process_calibration(self):
        await self.send_progress("Estimating 2D & 3D pose")
        kpts2d, scores2d, self.kpts3d, scores3d = await inferencer(self.video_folder, self.yolo_model, self.pose_estimator,
                                                        self.pose_lifter, self.device)

        await self.send_progress("Loading camera settings")
        # TODO: カメラデバイスに応じたパラメータ変更を可能にする
        param_iphone11 = np.load("./intrinsic/iphone11_4K.npz")
        param_iphone13 = np.load("./intrinsic/iphone13_4K.npz")
        self.K = np.array([param_iphone11["mtx"], param_iphone13["mtx"], param_iphone13["mtx"]])

        await self.send_progress("Processing calibration")
        self.R_est, self.t_est, kpts3d_est, self.kpts3d_tri = await calibrate(kpts2d, scores2d, self.kpts3d, scores3d, self.K)

    async def generate_result_video(self):
        whole3d = True
        if whole3d:
            kpts2d_dwp, scores2d_dwp = await inferencer_dwp(self.video_folder, self.yolo_model, self.dwp_estimator)
            kpts3d_tri =  triangulate_with_conf(kpts2d_dwp, scores2d_dwp, 
                                            self.K, self.R_est, self.t_est, (scores2d_dwp > 0))

        await self.send_progress("Generating result video")
        save_path = "./sample_output/demodemo.mp4"
        ani = vis_calib_res(kpts3d_tri, self.kpts3d, whole3d)
        ani.save(save_path, writer="ffmpeg")
        print("save mp4 finished")

        return save_path
    
    def update_video_folder(self, new_video_folder):
        self.video_folder = new_video_folder
        print(f"Video folder updated to: {new_video_folder}")

async def calibrate_video(video_folder, websocket, client_id):
    processor = VideoProcessor(video_folder, websocket, client_id)
    await processor.init_async()
    await processor.process_calibration()
    
    return processor

async def generate_video(processor):
    save_path = await processor.generate_result_video()

    return save_path