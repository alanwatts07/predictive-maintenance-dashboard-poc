# Predictive Maintenance Terminal

A real-time bearing health monitoring dashboard that visualizes equipment degradation through vibration analysis. Built for industrial IoT applications to enable predictive maintenance and prevent unexpected machinery failures.

![Dashboard Preview](./docs/preview.png)

## Live Demo

[View Live Demo](https://frontend-two-ashy-yla6dtp7w4.vercel.app)

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

### Backend (Optional)
- **FastAPI** - Modern Python web framework
- **WebSocket** - Real-time bidirectional communication
- **NumPy/SciPy** - Scientific computing for signal processing

## Quick Start

### Frontend Only (Demo Mode)

The dashboard runs entirely in the browser with simulated data - no backend required.

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Full Stack (With Backend)

1. Start the backend:
```bash
cd backend
pip install -r requirements.txt
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

### Training Data: NASA Bearing Dataset

This project's degradation model is informed by the [NASA IMS Bearing Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository), a benchmark dataset for predictive maintenance research containing run-to-failure vibration data from accelerated bearing tests.

**Dataset Characteristics:**
- 4 Rexnord ZA-2115 bearings under constant load (6000 lbs) and speed (2000 RPM)
- Vibration data sampled at 20 kHz, recorded in 1-second snapshots
- Test run until bearing failure (~35 days of continuous operation)
- Ground truth failure modes: inner race defect, roller element defect, outer race defect

### Feature Engineering

The simulation models key statistical features used in bearing health assessment:

| Feature | Description | Failure Indicator |
|---------|-------------|-------------------|
| **RMS Amplitude** | Root mean square of vibration signal | Increases 2-5x before failure |
| **Kurtosis** | Signal peakedness (shock detection) | Spikes indicate impact events |
| **Crest Factor** | Peak-to-RMS ratio | Elevated during early-stage defects |
| **Spectral Energy** | Frequency band power distribution | Shifts toward defect frequencies |

### Degradation Model

The current implementation uses a **physics-informed simulation** that replicates observed failure progression:

```
HEALTHY → DEGRADING → CRITICAL → FAILURE
   │          │           │
   └──────────┴───────────┴── RMS amplitude increases
                              Volatility (kurtosis proxy) increases
                              Shock events become more frequent
```

**State Transition Logic:**
- **HEALTHY**: Baseline RMS (~0.5g), low volatility
- **DEGRADING**: RMS drift (+0.01g/sec), increasing volatility
- **CRITICAL**: Rapid RMS escalation (+0.05g/sec), shock spikes

### Future: ML Model Integration

For production deployment, the simulation can be replaced with trained models:

- **LSTM/GRU Networks**: Sequence modeling for RUL (Remaining Useful Life) prediction
- **1D-CNN**: Pattern recognition on raw vibration waveforms
- **Isolation Forest**: Unsupervised anomaly detection for novel failure modes
- **XGBoost Classifier**: State classification using engineered features

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
│   │   ├── hooks/
│   │   │   └── useDataGenerator.ts  # Client-side data simulation
│   │   ├── page.tsx                 # Main dashboard
│   │   └── layout.tsx               # App layout
│   ├── package.json
│   └── vercel.json
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI server
│   │   └── generator.py             # Vibration data generator
│   └── requirements.txt
│
└── DATA_INTEGRATION.md              # MQTT integration guide
```

## License

MIT
