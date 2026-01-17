# Data Integration Guide

This dashboard currently runs on a **Synthetic Data Generator** (`backend/app/generator.py`). To use it with real predictive maintenance data, follow this guide.

## 1. Architecture Overview

**Current:**
`[Random Noise Generator] -> [FastAPI WebSocket] -> [React Frontend]`

**Production:**
`[Sensors] -> [MQTT Broker] -> [FastAPI Subscriber] -> [React Frontend]`

## 2. Setting up the Data Source

You will need a data stream. The industry standard is **MQTT**.

### Step A: The Sensor Layer
- **Hardware**: Piezoelectric Accelerometers (e.g., Wilcoxon, PCB).
- **Gateway**: connect sensors to an Edge Gateway (e.g., Advantech ADAM-6000 or Moxa) to digitize the 4-20mA signal.
- **Protocol**: Configure the Gateway to publish JSON to an MQTT Broker.

### Step B: The Backend Listener
Modify `backend/app/main.py` to listen to the MQTT topic instead of generating numbers.

**Prerequisites:**
```bash
pip install paho-mqtt
```

**Code Changes (`backend/app/main.py`):**

```python
import paho.mqtt.client as mqtt
import json

# MQTT Config
BROKER = "mqtt.yourplant.com"
TOPIC = "plant/bearing/+"

# ... existing FastAPI code ...

# MQTT Callback
def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    
    # Transform to our format
    processed_data = {
        "time": payload['timestamp'], # Unix timestamp in seconds
        "close": payload['rms_value'],
        "state": "HEALTHY" if payload['rms_value'] < 0.5 else "CRITICAL"
    }
    
    # Broadcast to frontend via existing WebSocket manager
    asyncio.run(manager.broadcast(json.dumps(processed_data)))

# Start Client
client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.subscribe(TOPIC)
client.loop_start() 
```

## 3. Recommended Hardware Stack
For a Proof of Concept (POC) demonstration, you don't need a factory. You can use:
1.  **Arduino Nicla Sense ME**: A tiny Bosch sensor board with high-quality accelerometers.
2.  **Raspberry Pi**: Running Mosquitto (MQTT Broker).
3.  **This Dashboard**: Running on a laptop connected to the Pi.

## 4. Scaling
For historical analysis (Backtesting), you should save the MQTT messages to a Time-Series Database:
- **InfluxDB** or **TimescaleDB**.
- Connect Grafana to that DB for long-term trends (weeks/months).
- Use **This Dashboard** for the "Operator View" (Live/Real-time).
