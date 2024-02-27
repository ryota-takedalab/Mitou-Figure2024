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