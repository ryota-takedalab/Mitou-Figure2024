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
import shutil


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

def delete_directory(dir_path: str):
    try:
        shutil.rmtree(dir_path)
        print(f"Deleted directory: {dir_path}")
    except Exception as e:
        print(f"Error deleting directory {dir_path}: {e}")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    if client_id not in active_connections:
        active_connections[client_id] = {}
    active_connections[client_id]['websocket'] = websocket


    try:
        while True:
            data = await websocket.receive_text()
            if data == 'ping':
                await websocket.send_text('pong')
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        if client_id in active_connections:
            if 'temp_folder_path' in active_connections[client_id]:
                delete_directory(active_connections[client_id]['temp_folder_path'])
            del active_connections[client_id]
        await websocket.close()


@app.post("/calibrate/{client_id}")
async def calibrate(client_id: str, files: List[UploadFile] = File(...)):
    upload_folder = "uploads"
    temp_folder_path = os.path.join(upload_folder, client_id)
    os.makedirs(temp_folder_path, exist_ok=True)

    if client_id not in active_connections:
        active_connections[client_id] = {}
    active_connections[client_id]['temp_folder_path'] = temp_folder_path

    try:
        for file in files:
            file_location = os.path.join(temp_folder_path, file.filename)
            async with aio_open(file_location, "wb") as file_object:
                while content := await file.read(1024):
                    await file_object.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

    processor = await run.calibrate_video(temp_folder_path, active_connections[client_id]['websocket'], client_id)
    active_connections[client_id]['processor'] = processor
    
    return {"message": "Calibration started, check WebSocket for progress"}


@app.post("/generate_video/{client_id}")
async def generate(client_id: str, files: List[UploadFile] = File(...)):
    if client_id not in active_connections or 'processor' not in active_connections[client_id]:
        raise HTTPException(status_code=404, detail="Processor not found. Please complete calibration first.")

    processor = active_connections[client_id]['processor']

    # 新しい動画ファイル群を扱うための新しい一時フォルダを作成
    new_video_folder = os.path.join(active_connections[client_id]['temp_folder_path'], "new_videos")
    os.makedirs(new_video_folder, exist_ok=True)

    # new_videos フォルダ内の既存ファイルをクリア
    for filename in os.listdir(new_video_folder):
        file_path = os.path.join(new_video_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete {filename}: {e}")

    for file in files:
        new_video_file_path = os.path.join(new_video_folder, file.filename)
        async with aio_open(new_video_file_path, "wb") as file_object:
            while content := await file.read(1024):
                await file_object.write(content)

    # VideoProcessorインスタンスのvideo_folderを新しいフォルダに更新
    processor.update_video_folder(new_video_folder)

    save_path = await run.generate_video(processor)

    return FileResponse(save_path, media_type='video/mp4', filename=Path(save_path).name)