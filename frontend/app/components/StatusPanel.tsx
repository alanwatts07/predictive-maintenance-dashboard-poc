'use client';

import React, { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle, Zap } from 'lucide-react';

interface VibrationData {
    time: number;
    state: string;
    degradation: number;
    close: number;
}

export const StatusPanel = () => {
    const [status, setStatus] = useState<VibrationData | null>(null);

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setStatus(data);
        };

        return () => ws.close();
    }, []);

    const getStatusMessage = (state: string) => {
        switch (state) {
            case 'HEALTHY': return "ALL SYSTEMS NOMINAL. BASELINE VIBRATION LEVELS.";
            case 'DEGRADING': return "WARNING: LUBRICATION BREAKDOWN DETECTED. PREDICTIVE FAULT IMMINENT.";
            case 'CRITICAL': return "CRITICAL FAILURE: OUTER RACE DAMAGED. IMMEDIATE SHUTDOWN RECOMMENDED.";
            default: return "INITIALIZING...";
        }
    };

    const getStatusColor = (state: string = "HEALTHY") => {
        if (state === "HEALTHY") return "text-green-500 border-green-500/30 bg-green-900/10";
        if (state === "DEGRADING") return "text-yellow-500 border-yellow-500/30 bg-yellow-900/10 animate-pulse";
        return "text-red-500 border-red-500/30 bg-red-900/10 animate-pulse";
    };

    return (
        <div className={`p-6 rounded-lg border ${getStatusColor(status?.state)} transition-all duration-300`}>
            <div className="flex items-center gap-4 mb-4">
                {status?.state === 'HEALTHY' && <CheckCircle className="w-8 h-8" />}
                {status?.state === 'DEGRADING' && <Zap className="w-8 h-8" />}
                {status?.state === 'CRITICAL' && <AlertCircle className="w-8 h-8" />}

                <div>
                    <h2 className="text-sm font-bold opacity-70 uppercase tracking-widest">System Status</h2>
                    <div className="text-2xl font-black tracking-tighter">{status?.state || "CONNECTING"}</div>
                </div>
            </div>

            <div className="font-mono text-sm leading-relaxed opacity-90 border-t border-current pt-4 mt-2 border-opacity-20">
                &gt; {getStatusMessage(status?.state || "")}
                <br />
                &gt; CURRENT RMS: {status?.close.toFixed(4)} g
                <br />
                &gt; FAULT PROBABILITY: {((status?.degradation || 0) * 100).toFixed(1)}%
            </div>
        </div>
    );
};
