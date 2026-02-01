# Predictive Maintenance Terminal

A real-time bearing health monitoring dashboard that visualizes equipment degradation through vibration analysis. Built for industrial IoT applications to enable predictive maintenance and prevent unexpected machinery failures.

![Dashboard Preview](./docs/preview.png)

## Live Demo

**Frontend Dashboard:** [frontend-two-ashy-yla6dtp7w4.vercel.app](https://frontend-two-ashy-yla6dtp7w4.vercel.app)

**ML Backend API:** [predictive-maintenance-api-production-e4fc.up.railway.app](https://predictive-maintenance-api-production-e4fc.up.railway.app/health)

## Features

- **Real-time Visualization**: Live candlestick charts displaying vibration RMS data at 1-second intervals
- **Health State Monitoring**: Automatic classification into HEALTHY, DEGRADING, and CRITICAL states
- **Fault Probability**: Continuous degradation percentage calculation based on vibration patterns
- **Multi-Bearing Support**: Monitor multiple bearing types (Rexnord, SKF, Timken) across test sets
- **Interactive Simulation**: Inject fault scenarios to demonstrate predictive capabilities
- **SMA Indicator**: 10-period Simple Moving Average overlay for trend analysis

## Tech Stack

### Frontend
- **Next.js 16** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Lightweight Charts** - High-performance financial charting library
- **Lucide React** - Icon library

### Backend
- **FastAPI** - Modern Python web framework
- **WebSocket** - Real-time bidirectional communication
- **NumPy/SciPy** - Scientific computing for signal processing

### ML Pipeline
- **XGBoost** - Gradient boosting classifier for health state prediction
- **scikit-learn** - Feature scaling and model evaluation
- **joblib** - Model serialization for production inference

## Quick Start

### Frontend Only (Demo Mode)

The dashboard runs entirely in the browser with simulated data - no backend required.

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Full Stack (With ML Backend)

1. Install dependencies and train the model:
```bash
cd backend
pip install -r requirements.txt

# Train the bearing health classifier
python -m ml.train_model
```

2. Start the backend:
```bash
uvicorn app.main:app --reload
```

2. Start the frontend with backend URL:
```bash
cd frontend
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

## Deployment

### Vercel (Recommended)

The frontend is configured for one-click Vercel deployment:

1. Push to GitHub
2. Import project in Vercel
3. Deploy (no configuration needed)

The demo mode works out of the box without any backend.

### With Backend

For production with real sensor data:

1. Deploy backend to Railway, Render, or Fly.io
2. Set `NEXT_PUBLIC_BACKEND_URL` environment variable in Vercel
3. Deploy frontend to Vercel

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (Next.js)                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Chart       │  │ Status      │  │ Simulation          │  │
│  │ Component   │  │ Panel       │  │ Controls            │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          │                                   │
│              ┌───────────▼───────────┐                       │
│              │   Data Generator      │ ← Demo Mode           │
│              │   (Client-side)       │   (No backend)        │
│              └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Optional: WebSocket
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend (Optional)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ WebSocket   │  │ Data        │  │ MQTT Client         │  │
│  │ Server      │  │ Generator   │  │ (Production)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Simulation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Normal** | Baseline vibration levels (~0.5g RMS) | Healthy equipment operation |
| **Lubrication Fault** | Gradual RMS increase with volatility | Early-stage bearing wear |
| **Outer Race Failure** | Rapid degradation with shock spikes | Critical failure scenario |
| **Demo Mode** | Automated progression through states | Presentation/demo |

## ML Methodology

### Model Architecture

**Algorithm**: XGBoost Gradient Boosting Classifier
- 100 estimators with max depth 6
- Multi-class softmax objective (`multi:softprob`)
- Early stopping with 10 rounds patience
- StandardScaler preprocessing for feature normalization

**Classification Output**:
```
HEALTHY   → Normal operation, no intervention needed
DEGRADING → Early-stage fault, schedule maintenance
CRITICAL  → Imminent failure, immediate action required
```

### Training Data: NASA IMS Bearing Dataset

Trained on the [NASA IMS Bearing Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository), a benchmark for predictive maintenance research.

**Dataset Characteristics:**
- 4 Rexnord ZA-2115 bearings under constant load (6000 lbs) at 2000 RPM
- Vibration sampled at 20 kHz (20,480 samples per second)
- Run-to-failure tests (~35 days continuous operation)
- Ground truth failures: inner race, roller element, outer race defects

**Labeling Strategy** (based on run-to-failure position):
```
├─────────── 70% ───────────┼──── 20% ────┼── 10% ──┤
│         HEALTHY           │  DEGRADING  │ CRITICAL│
└───────────────────────────┴─────────────┴─────────┘
        Time → Failure
```

### Feature Engineering (14 Features)

**Time Domain Features:**

| Feature | Formula | What It Detects |
|---------|---------|-----------------|
| RMS | √(mean(x²)) | Overall vibration energy |
| Peak | max(\|x\|) | Maximum amplitude |
| Crest Factor | Peak / RMS | Impulsive events |
| Kurtosis | E[(x-μ)⁴]/σ⁴ | Shock spikes (bearing impacts) |
| Skewness | E[(x-μ)³]/σ³ | Signal asymmetry |
| Std | σ | Vibration spread |
| Clearance Factor | Peak / mean(√\|x\|)² | Early-stage faults |
| Shape Factor | RMS / mean(\|x\|) | Waveform changes |
| Impulse Factor | Peak / mean(\|x\|) | Impact severity |

**Frequency Domain Features:**

| Feature | Description |
|---------|-------------|
| Spectral Centroid | Center of mass of frequency spectrum |
| Spectral Spread | Variance around spectral centroid |
| Band Energy (0-2kHz) | Low frequency content ratio |
| Band Energy (2-5kHz) | Mid frequency content ratio |
| Band Energy (5-10kHz) | High frequency defect signatures |

### Model Performance

Typical results on held-out test set:
- **Accuracy**: ~92-95%
- **F1 Score (macro)**: ~0.90
- **Confusion Matrix**: Strong diagonal with minimal HEALTHY↔CRITICAL confusion

### Live API Endpoints

The trained model is deployed on Railway:

```bash
# Check model status
curl https://predictive-maintenance-api-production-e4fc.up.railway.app/model/status

# Health check
curl https://predictive-maintenance-api-production-e4fc.up.railway.app/health

# Predict from raw signal (needs 1000+ samples)
curl -X POST https://predictive-maintenance-api-production-e4fc.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [...], "sample_rate": 20000}'

# WebSocket for real-time streaming
wss://predictive-maintenance-api-production-e4fc.up.railway.app/ws/real
```

### Retraining the Model

```bash
cd backend

# Option 1: Download NASA dataset and train
python -m ml.download_nasa_data
python -m ml.train_model

# Option 2: Train on synthetic data (faster, no download)
python -m ml.train_model  # Falls back to synthetic if no NASA data
```

Model artifacts saved to `backend/ml/models/`:
- `bearing_classifier.joblib` - Trained XGBoost model
- `feature_scaler.joblib` - StandardScaler for inference
- `model_metadata.json` - Training metrics and config

## Data Model

Each data point follows OHLC (Open-High-Low-Close) format:

```typescript
interface VibrationData {
  time: number;        // Unix timestamp
  open: number;        // RMS at interval start (g)
  high: number;        // Peak RMS in interval (g)
  low: number;         // Minimum RMS in interval (g)
  close: number;       // RMS at interval end (g)
  state: string;       // HEALTHY | DEGRADING | CRITICAL
  degradation: number; // 0.0 - 1.0 fault probability
}
```

## Real-World Integration

For production deployment with actual sensors, see [DATA_INTEGRATION.md](./DATA_INTEGRATION.md) for MQTT integration guide.

Supported sensor protocols:
- MQTT over WebSocket
- Direct WebSocket connections
- REST API polling (for lower-frequency data)

## Project Structure

```
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   ├── ChartComponent.tsx   # Real-time candlestick chart
│   │   │   └── StatusPanel.tsx      # Health status display
│   │   ├── page.tsx                 # Main dashboard
│   │   └── layout.tsx               # App layout with OG tags
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI server with WebSocket
│   │   ├── generator.py             # Simulation data generator
│   │   └── predictor.py             # ML inference wrapper
│   ├── ml/
│   │   ├── train_model.py           # XGBoost training pipeline
│   │   ├── feature_extraction.py    # 14-feature extraction
│   │   ├── generate_synthetic_data.py
│   │   ├── download_nasa_data.py    # NASA dataset downloader
│   │   └── models/
│   │       ├── bearing_classifier.joblib  # Trained model
│   │       ├── feature_scaler.joblib      # Feature scaler
│   │       └── model_metadata.json        # Training metrics
│   ├── requirements.txt
│   └── Procfile                     # Railway deployment config
│
└── DATA_INTEGRATION.md              # MQTT integration guide
```

## License

MIT
