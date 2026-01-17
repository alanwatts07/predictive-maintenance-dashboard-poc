'use client';

import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData } from 'lightweight-charts';
import React, { useEffect, useRef, useState } from 'react';

// Types for our data
interface VibrationData {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    state: string;
    degradation: number;
}

export const ChartComponent = () => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const [lastData, setLastData] = useState<VibrationData | null>(null);
    const [connectionStatus, setConnectionStatus] = useState("Disconnected");

    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const smaSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
    const lastTimeRef = useRef<number>(0);

    // SMA State
    const closePricesRef = useRef<number[]>([]);

    // Debug state
    const [debugLog, setDebugLog] = useState<string[]>([]);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Delay initialization to ensure DOM has computed dimensions
        const initTimer = setTimeout(() => {
            if (!chartContainerRef.current) return;

            // Initial Dimensions
            const rect = chartContainerRef.current.getBoundingClientRect();
            const width = rect.width || 800;
            const height = 400;

            setDebugLog(prev => [`Chart Init: ${Math.floor(width)}px x ${height}px`, ...prev]);

            // Initialize Chart
            const chart = createChart(chartContainerRef.current, {
                layout: {
                    background: { type: ColorType.Solid, color: '#1e1e1e' }, // Lighter gray for visibility check
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
                    barSpacing: 10,
                },
                crosshair: {
                    mode: 0,
                },
            });

            // Use v4 API
            const newSeries = chart.addCandlestickSeries({
                upColor: '#00ff41',
                downColor: '#ff3131',
                borderVisible: false,
                wickUpColor: '#00ff41',
                wickDownColor: '#ff3131',
            });

            // Add SMA Line
            const shaLine = chart.addLineSeries({
                color: '#2962FF',
                lineWidth: 2,
                title: 'SMA(10)',
                crosshairMarkerVisible: false,
            });

            chartRef.current = chart;
            seriesRef.current = newSeries;
            smaSeriesRef.current = shaLine;

            // Resize Observer to handle responsive layout
            const resizeObserver = new ResizeObserver(entries => {
                if (entries.length === 0 || !entries[0].target) return;

                // Use getBoundingClientRect for more accuracy with sub-pixel rendering
                const newRect = entries[0].target.getBoundingClientRect();

                if (newRect.width > 0 && newRect.height > 0) {
                    chart.applyOptions({ width: newRect.width, height: newRect.height });
                }
            });

            resizeObserver.observe(chartContainerRef.current);

            // Backup Window Resize Listener (Double safety)
            const handleWindowResize = () => {
                if (chartContainerRef.current) {
                    const newRect = chartContainerRef.current.getBoundingClientRect();
                    chart.applyOptions({ width: newRect.width, height: 400 });
                }
            };
            window.addEventListener('resize', handleWindowResize);

            // WebSocket Connection
            const ws = new WebSocket('ws://localhost:8000/ws');

            ws.onopen = () => {
                setConnectionStatus("Connected (Stream Active)");
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                // Lightweight charts expects seconds for time (Unix Timestamp)
                let time = Math.floor(data.time / 1000);

                // Prevent duplicate or old timestamps (chart crash prevention)
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

                // --- SMA Logic ---
                closePricesRef.current.push(data.close);
                if (closePricesRef.current.length > 20) {
                    closePricesRef.current.shift(); // keep size manageable
                }

                const period = 10;
                if (closePricesRef.current.length >= period) {
                    const slice = closePricesRef.current.slice(-period);
                    const avg = slice.reduce((a, b) => a + b, 0) / period;

                    shaLine.update({
                        time: time as any,
                        value: avg,
                    });
                }
                // ----------------

                // Debug Output
                setDebugLog(prev => [`${new Date().toLocaleTimeString()} ${JSON.stringify(candlestick)}`, ...prev].slice(0, 5));

                // Force scroll to latest data
                if (chartRef.current) {
                    chartRef.current.timeScale().scrollToPosition(0, false);
                }

                setLastData(data);
            };

            ws.onclose = () => setConnectionStatus("Disconnected");

            // Cleanup
            return () => {
                window.removeEventListener('resize', handleWindowResize);
                resizeObserver.disconnect();
                chart.remove();
                ws.close();
            }
        }, 100);

        return () => clearTimeout(initTimer);
    }, []);

    // Helper to color code state
    const getStateColor = (state: string) => {
        if (state === "HEALTHY") return "text-crypto-up";
        if (state === "DEGRADING") return "text-yellow-400";
        return "text-crypto-down animate-pulse";
    }

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
                    <div className="w-full bg-gray-800 rounded-full h-2.5 mt-2">
                        <div
                            className="bg-red-600 h-2.5 rounded-full transition-all duration-500"
                            style={{ width: `${(lastData?.degradation || 0) * 100}%` }}
                        ></div>
                    </div>
                </div>
            </div>

            {/* Chart */}
            <div className="relative w-full h-[400px] border border-gray-800 rounded-lg overflow-hidden bg-black">
                <div ref={chartContainerRef} className="absolute inset-0 w-full h-full" />
                <div className="absolute top-2 left-2 text-xs text-gray-500 pointer-events-none">
                    Real-time (1s Candles)
                </div>
            </div>

            {/* Debug Console */}
            <div className="p-2 bg-gray-900 rounded text-[10px] font-mono text-gray-400 h-24 overflow-y-auto w-full">
                <p className="font-bold text-gray-300 border-b border-gray-700 mb-1">DEBUG LOG (Show this to support)</p>
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
