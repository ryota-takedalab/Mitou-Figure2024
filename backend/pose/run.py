import numpy as np
import torch
import torch.nn as nn

from mmdet.apis import init_detector
from mmpose.apis import init_model
from mmpose.utils import adapt_mmdet_pipeline

from calib.calibration import calibrate
from inference import inferencer
from MotionAGFormer.model import MotionAGFormer
from vis.calibpose import vis_calib_res


def load_models(device):
    det_config = './mmpose/mmdet_cfg/rtmdet_m_640-8xb32_coco-person.py'
    det_ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    detector = init_detector(det_config, det_ckpt, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_config = './mmpose/mmpose_cfg/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
    pose_ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
    pose_estimator = init_model(pose_config, pose_ckpt, device=device)

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

    return detector, pose_estimator, pose_lifter


def main():
    # 3視点からの動画含まれるフォルダ
    video_folder = "./sample_video"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector, pose_estimator, pose_lifter = load_models(device)
    kpts2d, scores2d, kpts3d, scores3d = inferencer(video_folder,
                                                    detector, pose_estimator, pose_lifter, device)

    # カメラによって異なるので今後変更が必要
    param_c13 = np.load("./intrinsic/iphone13.npz")
    param_c2 = np.load("./intrinsic/iphone11pro.npz")
    K = np.array([param_c13["mtx"], param_c2["mtx"], param_c13["mtx"]])

    R_est, t_est, kpts3d_est, kpts3d_tri = calibrate(kpts2d, scores2d, kpts3d, scores3d, K)
    ani = vis_calib_res(kpts3d_tri, kpts3d)

    save_path = "./sample_output/result.gif"
    ani.save(save_path, writer="pillow")


if __name__ == "__main__":
    main()
