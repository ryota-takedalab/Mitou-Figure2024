""""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import WebSocket
import cv2
import os
import subprocess
from pathlib import Path
from uuid import uuid4
from aiofiles import open as aio_open
from typing import List
import asyncio
import run
import sys
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 本番環境では適切なオリジンに変更する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket接続を管理する辞書
active_connections = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            # ここでクライアントからのメッセージに応じた処理を行う
    except Exception as e:
        del active_connections[client_id]
        await websocket.close()
        print(f"WebSocket connection closed: {e}")


def save_calibration_data(client_id, R_est, t_est, kpts3d_est, kpts3d_tri):
    calibration_data = {
        "R_est": R_est.tolist(), 
        "t_est": t_est.tolist(),
        "kpts3d_est": kpts3d_est.tolist(),
        "kpts3d_tri": kpts3d_tri.tolist(),
    }
    file_path = f"./calibration_data/{client_id}.json"
    with open(file_path, 'w') as f:
        json.dump(calibration_data, f)

@app.post("/calibration/{client_id}")
async def handle_upload(client_id: str, files: List[UploadFile] = File(...)):
    upload_folder = "calib_video"
    temp_folder_id = uuid4()
    temp_folder_path = os.path.join(upload_folder, str(temp_folder_id))
    os.makedirs(temp_folder_path, exist_ok=True)

    file_locations = []

    try:
        for file in files:
            file_location = os.path.join(temp_folder_path, file.filename)
            file_locations.append(file_location)
            async with aio_open(file_location, "wb") as file_object:
                while content := await file.read(1024):
                    await file_object.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

    websocket = active_connections.get(client_id)
    if not websocket:
        raise HTTPException(status_code=404, detail="WebSocket connection not found")

    detector, dwp_estimator, R_est, t_est, kpts3d_est, kpts3d_tri = await run.calib(temp_folder_path, websocket, client_id)
    # キャリブレーションパラメータを保存
    save_calibration_data(client_id, R_est, t_est, kpts3d_est, kpts3d_tri)
    print("Calibration data saved")

    return {"message": "Calibration data saved successfully"}


@app.post("/upload/{client_id}")
async def handle_upload(client_id: str, files: List[UploadFile] = File(...)):
    upload_folder = "uploads"
    temp_folder_id = uuid4()
    temp_folder_path = os.path.join(upload_folder, str(temp_folder_id))
    os.makedirs(temp_folder_path, exist_ok=True)

    file_locations = []

    try:
        for file in files:
            file_location = os.path.join(temp_folder_path, file.filename)
            file_locations.append(file_location)
            async with aio_open(file_location, "wb") as file_object:
                while content := await file.read(1024):
                    await file_object.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

    # WebSocket経由でプログレス更新を送信するためにclient_idを使用
    websocket = active_connections.get(client_id)
    if not websocket:
        raise HTTPException(status_code=404, detail="WebSocket connection not found")

    # 非同期にrun.mainを呼び出し、WebSocketとclient_idを渡す
    save_path = await run.main(temp_folder_path, websocket, client_id)
    print("save_path finished")

    return FileResponse(save_path, media_type='video/mp4', filename=Path(save_path).name)
"""

"""
import numpy as np
import torch
import torch.nn as nn
import json
import asyncio
import aiofiles
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
    det_config = './mmpose/mmdet_cfg/rtmdet_m_640-8xb32_coco-person.py'
    det_ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    detector = init_detector(det_config, det_ckpt, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

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

    return detector, pose_estimator, pose_lifter, dwp_estimator

async def calib(video_folder,websocket=None, client_id=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    await send_progress(10, websocket, client_id)

    detector, pose_estimator, pose_lifter, dwp_estimator = load_models(device)
    await send_progress(30, websocket, client_id)


    kpts2d, scores2d, kpts3d, scores3d = inferencer(video_folder,
                                                    detector, pose_estimator, pose_lifter, device)

    await send_progress(50, websocket, client_id)
    # カメラによって異なるので今後変更が必要
    param_c13 = np.load("./intrinsic/iphone13.npz")
    param_c2 = np.load("./intrinsic/iphone11pro.npz")
    # K = np.array([param_c13["mtx"], param_c2["mtx"], param_c13["mtx"]])
    K = np.array([param_c2["mtx"], param_c13["mtx"], param_c13["mtx"]])

    R_est, t_est, kpts3d_est, kpts3d_tri = calibrate(kpts2d, scores2d, kpts3d, scores3d, K)

    print("calibrate finished")

    return detector, dwp_estimator, R_est, t_est, kpts3d_est, kpts3d_tri


async def main(video_folder, websocket=None, client_id=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector, pose_estimator, pose_lifter, dwp_estimator = load_models(device)
    calibration_data_path = f"./calibration_data/{client_id}.json"
    async with aiofiles.open(calibration_data_path, 'r') as f:
        calibration_data = json.loads(await f.read())
    
    R_est = np.array(calibration_data['R_est'])
    t_est = np.array(calibration_data['t_est'])
    kpts3d_est = np.array(calibration_data['kpts3d_est'])
    kpts3d_tri = np.array(calibration_data['kpts3d_tri'])

    await send_progress(70, websocket, client_id)
    kpts2d_dwp, scores2d_dwp = inferencer_dwp(video_folder, detector, dwp_estimator)
    whole3d = triangulate_with_conf(kpts2d_dwp, scores2d_dwp,
                                    K, R_est, t_est, (scores2d_dwp > 0))

    ani = vis_calib_res(whole3d, kpts3d)
    save_path = "./sample_output/result.mp4"
    ani.save(save_path, writer="ffmpeg")
    print("save mp4 finished")

    return save_path


if __name__ == "__main__":
    main()
"""
"""
import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid'; // UUIDを生成するためのライブラリ
import Head from 'next/head';
import { time } from 'console';

const IndexPage = () => {
  const [videoSrc, setVideoSrc] = useState('');
  const [uploadedVideos, setUploadedVideos] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [client_id, setClientId] = useState(() => uuidv4()); // client_id を初期化時に1度だけ生成

  const fileInputRef = useRef<HTMLInputElement>(null);// ファイル入力要素への参照
  const calibrationFileInputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);


  const handleCalibrationButtonClick = () => {
    if (calibrationFileInputRef.current) {
      calibrationFileInputRef.current.click(); // 実際のファイル入力をトリガー
    }
  };

  const handleCalibrationFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
        const formData = new FormData();
        // 全てのファイルをformDataに追加
        for (let i = 0; i < event.target.files.length; i++) {
            formData.append(`files`, event.target.files[i]);
        }

        try {
            const response = await fetch(`${process.env.REACT_APP_UPLOAD_URL || 'http://localhost:8000'}/calibration/${client_id}`, {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            console.log('Calibration successful');
        } catch (error) {
            console.error('Error uploading the calibration file:', error);
        }
    }
};

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();// カスタムボタンクリックで実際のファイル入力をトリガー
    }
  };
  
  const setupWebSocket = () => {
    const websocket = new WebSocket(`${process.env.REACT_APP_WEBSOCKET_URL || 'ws://localhost:8000/ws'}/${client_id}`);
    wsRef.current = websocket;

    websocket.onopen = () => console.log('WebSocket Connection opened');
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received WebSocket message:', data);

      // 進捗更新以外のメッセージも処理できるように拡張
      switch (data.type) {
        case 'progress':
          setProgress(data.progress);
          break;
        case 'error':
          console.error('Error from server:', data.message);
          break;
        // 他のメッセージタイプにも対応可能
        default:
          console.log('Received unknown message type:', data);
      }
    };
    websocket.onerror = (error) => console.error('WebSocket Error:', error);
    websocket.onclose = (event) => {
      console.log('WebSocket Connection closed', event);
      // 自動的に再接続を試みる
      if (!event.wasClean) {
        console.log('Unexpected closure. Reconnecting WebSocket...');
        setupWebSocket();
      }
    };
  };

  useEffect(() => {
    setupWebSocket();

    // コンポーネントのクリーンアップ時にWebSocketを閉じる
    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting'); // Clean close
      }
    };
  }, [client_id]); 

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
        const formData = new FormData();
        const uploadedUrls = Array.from(event.target.files).map(file => {
            formData.append('files', file);
            return URL.createObjectURL(file); // アップロードされたファイルからURLを生成
        });

        setUploading(true);
        setProgress(0); // アップロード開始時に進捗をリセット
        setUploadedVideos(uploadedUrls); // アップロードされた動画のURLを状態に保存

        try {
          const response = await fetch(`${process.env.REACT_APP_UPLOAD_URL || 'http://localhost:8000/upload/'}${client_id}`, {
            method: 'POST',
            body: formData,
          });
          if (!response.ok) {
              throw new Error('Network response was not ok');
          }
          const blob = await response.blob();
          const videoUrl = URL.createObjectURL(blob);
          setVideoSrc(videoUrl); // 処理後の動画URLをセット
        } catch (error) {
          console.error('Error uploading the files:', error);
        } finally {
          setUploading(false);
        }
    }
};


return (
  <div>
    <Head>
      <title>Mitou 2024 Demo</title>
    </Head>
    <h1 style={{ fontFamily: "'Passion One', cursive" }}>Upload Videos for Pose Estimation</h1>
    <button onClick={handleCalibrationButtonClick} disabled={uploading} className="customButton">
        {uploading ? 'Processing...' : 'Calibrate'}
      </button>
      <input
        type="file"
        multiple
        ref={calibrationFileInputRef}
        onChange={handleCalibrationFileChange}
        style={{ display: 'none' }}
      />
    <button onClick={handleButtonClick} disabled={uploading} className="customButton">
      {uploading ? 'Uploading...' : 'Select Files'} {/* アップロード中はボタンのテキストを変更 */}
    </button>
    <input
      type="file"
      multiple
      ref={fileInputRef}
      onChange={handleFileChange}
      style={{ display: 'none' }} // 元のファイル入力は非表示
    />
    {uploading && (
      <div>
        <p>Uploading and processing... {progress}%</p>
        <progress value={progress} max="100"></progress>
      </div>
    )}
    {videoSrc && (
      <div style={{ textAlign: 'center'}}>
        <p style={{ fontSize: '50px' }}>Result Video</p>
        <video controls src={videoSrc} style={{width:"720px", margin: "0 auto 80px", display: 'block'}} autoPlay loop>
          Your browser does not support the video tag.
        </video>
      </div>
    )}
    <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap' }}>
      {uploadedVideos.map((videoSrc, index) => (
        <div key={index} style={{ margin: '10px', textAlign: 'center'}}> {/* 動画の間隔を設定 */}
          <p style={{ margin: "0 auto", fontSize: "30px"}}>{`Uploaded Video ${index + 1}`}</p> {/* 動画の番号を表示 */}
          <video controls src={videoSrc} style={{ width: "500px", display: 'block' }}></video>
        </div>
      ))}
    </div>
  </div>
 );
};

export default IndexPage;
"""