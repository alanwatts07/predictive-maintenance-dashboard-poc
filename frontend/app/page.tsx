'use client';

import { ChartComponent } from "./components/ChartComponent";
import { StatusPanel } from "./components/StatusPanel";
import { Activity, AlertTriangle, CheckCircle, Zap } from "lucide-react";

export default function Home() {

  const setControl = async (state: string) => {
    try {
      await fetch(`http://localhost:8000/control/${state}`, { method: 'POST' });
    } catch (e) {
      console.error("Failed to set control", e);
    }
  };

  return (
    <main className="min-h-screen p-8 flex flex-col gap-6 bg-[#050505] max-w-[1800px] mx-auto">
      {/* Top Bar */}
      <header className="flex justify-between items-center border-b border-gray-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded">
            <Activity className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">PRED-MAINT TERMINAL</h1>
            <p className="text-xs text-gray-400">NASA Bearing Dataset // Synthetic Stream</p>
          </div>
        </div>
        <div className="flex gap-4">
          <button className="px-4 py-1 text-xs font-mono bg-gray-900 border border-gray-700 rounded text-gray-400 hover:bg-gray-800 transition">
            EXP: RAW_DATA
          </button>
          <button className="px-4 py-1 text-xs font-mono bg-blue-900/30 border border-blue-800 text-blue-400 rounded hover:bg-blue-900/50 transition animate-pulse">
            LIVE: ON
          </button>
        </div>
      </header>

      {/* Chart Section - Full Width */}
      <div className="w-full">
        <ChartComponent />
      </div>

      {/* Bottom Grid: Status | Controls | Logistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* Col 1: Status Panel */}
        <StatusPanel />

        {/* Col 2: Controls */}
        <div className="bg-glass p-6 rounded-lg border border-gray-800">
          <h2 className="text-sm font-bold text-gray-400 uppercase mb-4">Simulation Controls</h2>
          <div className="flex flex-col gap-2">
            <button
              onClick={() => setControl('HEALTHY')}
              className="flex items-center gap-3 p-3 rounded bg-gray-800 border border-green-900/50 hover:bg-gray-700 hover:border-green-500 transition group"
            >
              <CheckCircle className="w-5 h-5 text-green-500 group-hover:text-green-400" />
              <div className="text-left">
                <div className="text-sm font-bold text-white">Normal Operation</div>
                <div className="text-[10px] text-gray-400">Inject baseline noise only</div>
              </div>
            </button>

            <button
              onClick={() => setControl('DEGRADING')}
              className="flex items-center gap-3 p-3 rounded bg-gray-800 border border-yellow-900/50 hover:bg-gray-700 hover:border-yellow-500 transition group"
            >
              <Zap className="w-5 h-5 text-yellow-500 group-hover:text-yellow-400" />
              <div className="text-left">
                <div className="text-sm font-bold text-white">Inject Lubrication Fault</div>
                <div className="text-[10px] text-gray-400">Increase volatility & RMS</div>
              </div>
            </button>

            <button
              onClick={() => setControl('CRITICAL')}
              className="flex items-center gap-3 p-3 rounded bg-gray-800 border border-red-900/50 hover:bg-gray-700 hover:border-red-500 transition group"
            >
              <AlertTriangle className="w-5 h-5 text-red-500 group-hover:text-red-400" />
              <div className="text-left">
                <div className="text-sm font-bold text-white">Inject Outer Race Failure</div>
                <div className="text-[10px] text-gray-400">Critical vibration logic</div>
              </div>
            </button>

            <div className="h-px bg-gray-800 my-2" />

            <button
              onClick={() => setControl('GRADUAL')}
              className="flex items-center gap-3 p-3 rounded bg-blue-900/10 border border-blue-900/30 hover:bg-blue-900/20 hover:border-blue-500 transition group"
            >
              <Activity className="w-5 h-5 text-blue-400 group-hover:text-blue-300 animate-pulse" />
              <div className="text-left">
                <div className="text-sm font-bold text-blue-400">Simulate Wear (Demo Mode)</div>
                <div className="text-[10px] text-gray-400">Gradually increase degradation over 30s</div>
              </div>
            </button>
          </div>
        </div>

        {/* Col 3: Logistics */}
        <div className="bg-glass p-6 rounded-lg border border-gray-800">
          <h2 className="text-sm font-bold text-gray-400 uppercase mb-4">System Logistics</h2>
          <div className="space-y-4 font-mono text-xs">
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-500">Bearing Type</span>
              <span className="text-white">Rexnord ZA-2115</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-500">Sampling Rate</span>
              <span className="text-white">20 kHz</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-500">RMS Threshold</span>
              <span className="text-red-400">0.9 g</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-500">Protocol</span>
              <span className="text-blue-400">MQTT over WSS</span>
            </div>
          </div>
        </div>

      </div>
    </main>
  );
}
