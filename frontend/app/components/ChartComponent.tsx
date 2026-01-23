'use client';

import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData } from 'lightweight-charts';
import React, { useEffect, useRef, useState, useCallback } from 'react';

// Types for our data
export interface VibrationData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  state: string;
  degradation: number;
}

export type SimulationState = 'HEALTHY' | 'DEGRADING' | 'CRITICAL' | 'GRADUAL';

interface ChartComponentProps {
  onDataUpdate?: (data: VibrationData) => void;
  onStateChange?: (setState: (state: SimulationState) => void) => void;
}

// Check if we have a backend URL configured
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || '';
const USE_WEBSOCKET = BACKEND_URL.length > 0;

export const ChartComponent: React.FC<ChartComponentProps> = ({ onDataUpdate, onStateChange }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [lastData, setLastData] = useState<VibrationData | null>(null);
  const [connectionStatus, setConnectionStatus] = useState("Initializing...");

  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const smaSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const lastTimeRef = useRef<number>(0);

  // SMA State
  const closePricesRef = useRef<number[]>([]);

  // Debug state
  const [debugLog, setDebugLog] = useState<string[]>([]);

  // Demo mode generator state
  const stateRef = useRef<SimulationState>('HEALTHY');
  const degradationFactorRef = useRef(0.0);
  const lastUpdateRef = useRef(Date.now());

  const BASE_RMS = 0.5;

  // Gaussian random number generator (Box-Muller transform)
  const gaussianRandom = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  // Generate simulated data chunk (demo mode)
  const generateChunk = useCallback((): VibrationData => {
    const now = Date.now();
    const dt = (now - lastUpdateRef.current) / 1000;
    lastUpdateRef.current = now;

    // Evolve state based on current mode
    if (stateRef.current === 'DEGRADING') {
      degradationFactorRef.current = Math.min(degradationFactorRef.current + 0.01 * dt, 0.7);
    } else if (stateRef.current === 'CRITICAL') {
      degradationFactorRef.current = Math.min(degradationFactorRef.current + 0.05 * dt, 2.0);
    } else if (stateRef.current === 'GRADUAL') {
      degradationFactorRef.current += 0.02 * dt;
    }

    // Determine reported state for GRADUAL mode
    let reportedState: string = stateRef.current;
    if (stateRef.current === 'GRADUAL') {
      if (degradationFactorRef.current < 0.3) {
        reportedState = 'HEALTHY';
      } else if (degradationFactorRef.current < 0.7) {
        reportedState = 'DEGRADING';
      } else {
        reportedState = 'CRITICAL';
      }
    }

    // Calculate target RMS based on degradation
    const currentRmsTarget = BASE_RMS + (degradationFactorRef.current * 2.5);
    const volatility = 0.05 + (degradationFactorRef.current * 0.2);

    const openVal = currentRmsTarget + gaussianRandom() * volatility;
    const closeVal = currentRmsTarget + gaussianRandom() * volatility;

    // Shock factor simulates bearing impacts
    const shockFactor = 1.0 + (degradationFactorRef.current * 3.0);
    const highVal = Math.max(openVal, closeVal) + Math.abs(gaussianRandom() * volatility * shockFactor);
    const lowVal = Math.max(0.01, Math.min(openVal, closeVal) - Math.abs(gaussianRandom() * volatility * shockFactor));

    return {
      time: now,
      open: parseFloat(openVal.toFixed(4)),
      high: parseFloat(highVal.toFixed(4)),
      low: parseFloat(lowVal.toFixed(4)),
      close: parseFloat(closeVal.toFixed(4)),
      state: reportedState,
      degradation: parseFloat(degradationFactorRef.current.toFixed(2)),
    };
  }, []);

  // Set simulation state (exposed to parent)
  const setSimulationState = useCallback((newState: SimulationState) => {
    stateRef.current = newState;

    if (newState === 'HEALTHY') {
      degradationFactorRef.current = 0.0;
    } else if (newState === 'DEGRADING') {
      degradationFactorRef.current = 0.3;
    } else if (newState === 'CRITICAL') {
      degradationFactorRef.current = 0.8;
    } else if (newState === 'GRADUAL') {
      degradationFactorRef.current = 0.0;
    }
  }, []);

  // Expose state change function to parent
  useEffect(() => {
    if (onStateChange) {
      onStateChange(setSimulationState);
    }
  }, [onStateChange, setSimulationState]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Delay initialization to ensure DOM has computed dimensions
    const initTimer = setTimeout(() => {
      if (!chartContainerRef.current) return;

      // Initial Dimensions
      const rect = chartContainerRef.current.getBoundingClientRect();
      const width = rect.width || 800;
      const height = 500;

      setDebugLog(prev => [`Chart Init: ${Math.floor(width)}px x ${height}px`, ...prev]);

      // Initialize Chart
      const chart = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: '#1e1e1e' },
          textColor: '#d1d4dc',
        },
        grid: {
          vertLines: { color: '#2b2b2b' },
          horzLines: { color: '#2b2b2b' },
        },
        width: width,
        height: height,
        timeScale: {
          timeVisible: true,
          secondsVisible: true,
          rightOffset: 10,
          barSpacing: 28,
        },
        crosshair: {
          mode: 0,
        },
      });

      const newSeries = chart.addCandlestickSeries({
        upColor: '#00ff41',
        downColor: '#ff3131',
        borderVisible: false,
        wickUpColor: '#00ff41',
        wickDownColor: '#ff3131',
      });

      const smaLine = chart.addLineSeries({
        color: '#2962FF',
        lineWidth: 2,
        title: 'SMA(10)',
        crosshairMarkerVisible: false,
      });

      chartRef.current = chart;
      seriesRef.current = newSeries;
      smaSeriesRef.current = smaLine;

      // Resize Observer
      const resizeObserver = new ResizeObserver(entries => {
        if (entries.length === 0 || !entries[0].target) return;
        const newRect = entries[0].target.getBoundingClientRect();
        if (newRect.width > 0 && newRect.height > 0) {
          chart.applyOptions({ width: newRect.width, height: newRect.height });
        }
      });

      resizeObserver.observe(chartContainerRef.current);

      const handleWindowResize = () => {
        if (chartContainerRef.current) {
          const newRect = chartContainerRef.current.getBoundingClientRect();
          chart.applyOptions({ width: newRect.width, height: 500 });
        }
      };
      window.addEventListener('resize', handleWindowResize);

      // Process incoming data (works for both WebSocket and demo mode)
      const processData = (data: VibrationData) => {
        let time = Math.floor(data.time / 1000);

        if (time <= lastTimeRef.current) {
          time = lastTimeRef.current + 1;
        }
        lastTimeRef.current = time;

        const candlestick: CandlestickData = {
          time: time as any,
          open: data.open,
          high: data.high,
          low: data.low,
          close: data.close,
        };

        newSeries.update(candlestick);

        // SMA Logic
        closePricesRef.current.push(data.close);
        if (closePricesRef.current.length > 20) {
          closePricesRef.current.shift();
        }

        const period = 10;
        if (closePricesRef.current.length >= period) {
          const slice = closePricesRef.current.slice(-period);
          const avg = slice.reduce((a, b) => a + b, 0) / period;
          smaLine.update({
            time: time as any,
            value: avg,
          });
        }

        setDebugLog(prev => [`${new Date().toLocaleTimeString()} ${JSON.stringify(candlestick)}`, ...prev].slice(0, 5));

        if (chartRef.current) {
          chartRef.current.timeScale().scrollToPosition(0, false);
        }

        setLastData(data);
        if (onDataUpdate) {
          onDataUpdate(data);
        }
      };

      let ws: WebSocket | null = null;
      let demoInterval: NodeJS.Timeout | null = null;

      if (USE_WEBSOCKET) {
        // WebSocket mode - connect to backend
        const wsUrl = BACKEND_URL.replace('http', 'ws') + '/ws';
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setConnectionStatus("Connected (Live Stream)");
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          processData(data);
        };

        ws.onclose = () => setConnectionStatus("Disconnected");
        ws.onerror = () => setConnectionStatus("Connection Error");
      } else {
        // Demo mode - client-side simulation
        setConnectionStatus("Demo Mode (Simulated Data)");

        // Generate initial data
        processData(generateChunk());

        // Generate data every second
        demoInterval = setInterval(() => {
          processData(generateChunk());
        }, 1000);
      }

      // Cleanup
      return () => {
        window.removeEventListener('resize', handleWindowResize);
        resizeObserver.disconnect();
        chart.remove();
        if (ws) ws.close();
        if (demoInterval) clearInterval(demoInterval);
      };
    }, 100);

    return () => clearTimeout(initTimer);
  }, [generateChunk, onDataUpdate]);

  // Helper to color code state
  const getStateColor = (state: string) => {
    if (state === "HEALTHY") return "text-crypto-up";
    if (state === "DEGRADING") return "text-yellow-400";
    return "text-crypto-down animate-pulse";
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Ticker Header */}
      <div className="grid grid-cols-4 gap-4 p-4 bg-glass rounded-lg w-full">
        <div>
          <p className="text-gray-500 text-xs uppercase">Symbol</p>
          <p className="text-xl font-bold text-white">BRG-001</p>
        </div>
        <div>
          <p className="text-gray-500 text-xs uppercase">Vibration RMS</p>
          <p className={`text-2xl font-mono ${lastData?.close! > lastData?.open! ? 'text-crypto-up' : 'text-crypto-down'}`}>
            {lastData?.close.toFixed(4) || "0.0000"} g
          </p>
        </div>
        <div>
          <p className="text-gray-500 text-xs uppercase">Machine Health</p>
          <p className={`text-xl font-bold ${getStateColor(lastData?.state || "HEALTHY")}`}>
            {lastData?.state || "Waiting..."}
          </p>
        </div>
        <div>
          <p className="text-gray-500 text-xs uppercase">Fault Prob.</p>
          <div className="mt-1">
            <p className={`text-2xl font-mono font-bold ${getStateColor(lastData?.state || "HEALTHY")}`}>
              {((lastData?.degradation || 0) * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="relative w-full h-[500px] border border-gray-800 rounded-lg overflow-hidden bg-black">
        <div ref={chartContainerRef} className="absolute inset-0 w-full h-full" />
        <div className="absolute top-2 left-2 text-xs text-gray-500 pointer-events-none">
          Real-time (1s Candles)
        </div>
      </div>

      {/* Debug Console */}
      <div className="p-2 bg-gray-900 rounded text-[10px] font-mono text-gray-400 h-24 overflow-y-auto w-full">
        <p className="font-bold text-gray-300 border-b border-gray-700 mb-1">DEBUG LOG</p>
        <div className="flex flex-col-reverse">
          {debugLog.map((log, i) => (
            <div key={i}>{log}</div>
          ))}
        </div>
      </div>

      <div className="text-xs text-gray-600 font-mono">
        Status: {connectionStatus}
      </div>
    </div>
  );
};
