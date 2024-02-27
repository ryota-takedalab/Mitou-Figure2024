import asyncio

async def send_progress(progress, websocket=None, client_id=None):
    if websocket and client_id:
        print(f"Sending progress {progress} to client {client_id}")
        await websocket.send_json({"client_id": client_id, "progress": progress})
        await asyncio.sleep(0) # イベントループにコントロールを戻す