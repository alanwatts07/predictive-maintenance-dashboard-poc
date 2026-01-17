from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from .generator import generator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Generate data
            data = generator.generate_chunk()
            await websocket.send_text(json.dumps(data))
            # Stream at 1Hz for clear 1s candles (Stock market style)
            await asyncio.sleep(1.0) 
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/control/{state}")
async def set_state(state: str):
    """
    Control the plant simulation.
    States: HEALTHY, DEGRADING, CRITICAL
    """
    state = state.upper()
    generator.set_state(state)
    return {"status": "ok", "current_state": generator.state}
