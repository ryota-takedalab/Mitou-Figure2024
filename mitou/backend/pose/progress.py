import asyncio

async def send_progress(message, websocket=None, client_id=None):
    if websocket and client_id:
        print(f"Sending status '{message}' to client {client_id}")
        await websocket.send_json({"client_id": client_id, "status": message})
        await asyncio.sleep(0)  # イベントループにコントロールを戻す