import numpy as np
import time
import asyncio
import random
from datetime import datetime

class DataGenerator:
    def __init__(self):
        self.state = "HEALTHY" # HEALTHY, DEGRADING, CRITICAL
        self.degradation_factor = 0.0
        self.base_rms = 0.5
        self.noise_level = 0.1
        self.last_update = time.time()
        
    def set_state(self, new_state: str):
        if new_state in ["HEALTHY", "DEGRADING", "CRITICAL", "GRADUAL"]:
            self.state = new_state
            if new_state == "HEALTHY":
                self.degradation_factor = 0.0
            elif new_state == "DEGRADING":
                self.degradation_factor = 0.3
            elif new_state == "CRITICAL":
                self.degradation_factor = 0.8
            elif new_state == "GRADUAL":
                self.degradation_factor = 0.0 # Start fresh

    def generate_chunk(self):
        """
        Simulate 1 second of vibration data and return statistics (Candlestick data).
        """
        # Evolve state
        dt = time.time() - self.last_update
        self.last_update = time.time()

        if self.state == "DEGRADING":
            self.degradation_factor = min(self.degradation_factor + 0.01 * dt, 0.7)
        elif self.state == "CRITICAL":
            self.degradation_factor = min(self.degradation_factor + 0.05 * dt, 2.0)
        elif self.state == "GRADUAL":
             # Slow creep
             self.degradation_factor += 0.02 * dt
             
        # Determine "Reported State" (what the UI shows) based on physical degradation
        reported_state = self.state
        if self.state == "GRADUAL":
            if self.degradation_factor < 0.3:
                reported_state = "HEALTHY"
            elif self.degradation_factor < 0.7:
                reported_state = "DEGRADING"
            else:
                reported_state = "CRITICAL"
        
        # Base signal parameters
        current_rms_target = self.base_rms + (self.degradation_factor * 2.5)
        
        # Random fluctuation for "Stock Market" feel
        volatility = 0.05 + (self.degradation_factor * 0.2)
        
        # Generate high-frequency synthetic raw data (conceptually)
        # We define Open-High-Low-Close for the "Candle" based on statistics
        
        open_val = current_rms_target + np.random.normal(0, volatility)
        close_val = current_rms_target + np.random.normal(0, volatility)
        
        # High/Low depend on how "shocky" the signal is (Kurtosis proxy)
        shock_factor = 1.0 + (self.degradation_factor * 3.0) 
        high_val = max(open_val, close_val) + abs(np.random.normal(0, volatility * shock_factor))
        low_val = min(open_val, close_val) - abs(np.random.normal(0, volatility * shock_factor))
        
        # Ensure positivity for physical magnitude
        low_val = max(0.01, low_val)
        
        timestamp = int(datetime.now().timestamp() * 1000)
        
        return {
            "time": timestamp / 1000, # Lightweight charts wants seconds usually, but we'll send epoch
            "open": round(open_val, 4),
            "high": round(high_val, 4),
            "low": round(low_val, 4),
            "close": round(close_val, 4),
            "state": reported_state,
            "degradation": round(self.degradation_factor, 2)
        }

generator = DataGenerator()
