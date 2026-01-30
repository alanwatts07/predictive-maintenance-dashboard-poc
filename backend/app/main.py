"""
FastAPI backend for Predictive Maintenance Dashboard.

Supports two modes:
1. SIMULATION (default): Generates synthetic data for demo
2. REAL: Uses trained ML model to analyze actual sensor data

Endpoints:
- WebSocket /ws: Stream data (simulation mode)
- WebSocket /ws/real: Stream predictions from real sensor data
- POST /control/{state}: Control simulation state
- POST /predict: One-shot prediction from vibration data
- GET /model/status: Check if ML model is loaded
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import numpy as np
from typing import Optional
from pydantic import BaseModel

from .generator import generator
from .predictor import get_predictor, is_model_ready

app = FastAPI(
    title="Predictive Maintenance API",
    description="Real-time bearing health monitoring with ML-based prediction",
    version="2.0.0"
)

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
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


# ============== Simulation Mode Endpoints ==============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for SIMULATION mode.
    Streams synthetic vibration data at 1Hz.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = generator.generate_chunk()
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/control/{state}")
async def set_state(state: str):
    """
    Control the simulation state.
    States: HEALTHY, DEGRADING, CRITICAL, GRADUAL
    """
    state = state.upper()
    if state not in ["HEALTHY", "DEGRADING", "CRITICAL", "GRADUAL"]:
        raise HTTPException(status_code=400, detail=f"Invalid state: {state}")
    generator.set_state(state)
    return {"status": "ok", "current_state": generator.state}


# ============== Real ML Mode Endpoints ==============

class VibrationData(BaseModel):
    """Request body for vibration data prediction."""
    signal: list[float]
    sample_rate: float = 20000


class FeatureData(BaseModel):
    """Request body for pre-computed features."""
    rms: float
    peak: float
    crest_factor: float
    kurtosis: float
    skewness: float
    std: float
    clearance_factor: float
    shape_factor: float
    impulse_factor: float
    spectral_centroid: Optional[float] = 0
    spectral_spread: Optional[float] = 0
    band_energy_low: Optional[float] = 0
    band_energy_mid: Optional[float] = 0
    band_energy_high: Optional[float] = 0


@app.get("/model/status")
async def model_status():
    """Check if the ML model is loaded and ready."""
    ready = is_model_ready()
    return {
        "model_loaded": ready,
        "mode": "real" if ready else "simulation_only",
        "message": "ML model ready for predictions" if ready else "ML model not loaded. Run train_model.py to enable real predictions."
    }


@app.post("/predict")
async def predict_from_signal(data: VibrationData):
    """
    Predict bearing health from raw vibration signal.

    Send a 1-second window of vibration data (e.g., 20000 samples at 20kHz)
    and receive health state prediction.
    """
    if not is_model_ready():
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run train_model.py first."
        )

    signal = np.array(data.signal)
    if len(signal) < 1000:
        raise HTTPException(
            status_code=400,
            detail=f"Signal too short: {len(signal)} samples. Need at least 1000."
        )

    predictor = get_predictor()
    result = predictor.predict_from_signal(signal, data.sample_rate)

    return result


@app.post("/predict/features")
async def predict_from_features(features: FeatureData):
    """
    Predict bearing health from pre-computed features.

    Useful when feature extraction is done on edge devices.
    """
    if not is_model_ready():
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run train_model.py first."
        )

    predictor = get_predictor()
    result = predictor.predict_from_features(features.model_dump())

    return result


@app.websocket("/ws/real")
async def websocket_real_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for REAL sensor data.

    Client sends vibration data, server returns predictions.

    Protocol:
    1. Client connects
    2. Client sends JSON: {"signal": [float, ...], "sample_rate": 20000}
    3. Server responds with prediction
    4. Repeat
    """
    if not is_model_ready():
        await websocket.close(code=4003, reason="ML model not loaded")
        return

    await websocket.accept()
    predictor = get_predictor()

    try:
        while True:
            # Receive vibration data from client
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)

            signal = np.array(data.get("signal", []))
            sample_rate = data.get("sample_rate", 20000)

            if len(signal) < 1000:
                await websocket.send_text(json.dumps({
                    "error": "Signal too short",
                    "received_samples": len(signal)
                }))
                continue

            # Run prediction
            result = predictor.predict_from_signal(signal, sample_rate)

            # Format for frontend (compatible with simulation format)
            from datetime import datetime
            timestamp = int(datetime.now().timestamp() * 1000)

            response = {
                "time": timestamp / 1000,
                "open": result["features"]["rms"],
                "high": result["features"]["peak"],
                "low": result["features"]["rms"] * 0.8,  # Approximate
                "close": result["features"]["rms"],
                "state": result["state"],
                "degradation": result["degradation"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "mode": "real"
            }

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))


# ============== Health Check ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ml_model_ready": is_model_ready(),
        "simulation_ready": True
    }
