from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
import subprocess
from pathlib import Path
from uuid import uuid4
from aiofiles import open as aio_open

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost"],  # 本番環境では適切なオリジンに変更する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def handle_upload(file: UploadFile = File(...)):
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    # ファイル名にユニークなIDを追加
    file_location = f"{upload_folder}/{uuid4()}_{file.filename}"

    try:
        # ファイルをチャンクで非同期に書き込む
        async with aio_open(file_location, "wb") as file_object:
            while content := await file.read(1024):  # チャンクサイズを1024バイトに設定
                await file_object.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

    grayscale_video_path = convert_to_grayscale(file_location)
    compatible_video_path = await convert_for_browser_compatibility(grayscale_video_path)

    return FileResponse(compatible_video_path, media_type='video/mp4', filename=Path(compatible_video_path).name)

def convert_to_grayscale(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = video_path.replace('.mp4', '_grayscale.mp4')
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), isColor=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray_frame)
    
    cap.release()
    out.release()
    return out_path

async def convert_for_browser_compatibility(video_path):
    output_path = video_path.replace('_grayscale.mp4', '_grayscale_compatible.mp4')
    cmd = [
        "ffmpeg", "-i", video_path, 
        "-c:v", "libx264", "-preset", "medium", "-b:v", "2500k", 
        "-c:a", "aac", "-b:a", "128k", 
        output_path
    ]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Video conversion failed: {process.stderr}")
    return output_path


