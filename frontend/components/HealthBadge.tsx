"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Activity, CheckCircle2, XCircle } from "lucide-react";

export function HealthBadge({ compact = false }: { compact?: boolean }) {
  const [status, setStatus] = useState<"checking" | "healthy" | "offline">("checking");
  const [latency, setLatency] = useState<number | null>(null);

  const checkStatus = async () => {
    const start = performance.now();
    try {
      const data = await api.checkHealth();
      const end = performance.now();
      if (data.status === "healthy") {
        setStatus("healthy");
        setLatency(Math.round(end - start));
      } else {
        setStatus("offline");
      }
    } catch {
      setStatus("offline");
      setLatency(null);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  if (compact) {
    return (
      <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-900/80 border border-slate-800 text-xs font-medium">
        <span
          className={`w-2 h-2 rounded-full ${
            status === "healthy"
              ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)] animate-pulse"
              : status === "offline"
              ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]"
              : "bg-amber-500 animate-spin"
          }`}
        />
        <span className="text-slate-300">
          {status === "healthy"
            ? `API Online ${latency ? `(${latency}ms)` : ""}`
            : status === "offline"
            ? "API Offline"
            : "Checking..."}
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80">
      <div
        className={`p-2.5 rounded-lg ${
          status === "healthy"
            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
            : status === "offline"
            ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
            : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
        }`}
      >
        {status === "healthy" ? (
          <CheckCircle2 className="w-5 h-5" />
        ) : status === "offline" ? (
          <XCircle className="w-5 h-5" />
        ) : (
          <Activity className="w-5 h-5 animate-spin" />
        )}
      </div>
      <div>
        <div className="flex items-center gap-2">
          <p className="text-sm font-semibold text-slate-100">
            NexusAI Backend API
          </p>
          <span
            className={`text-[10px] font-bold px-2 py-0.5 rounded-md uppercase tracking-wider ${
              status === "healthy"
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                : status === "offline"
                ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                : "bg-amber-500/20 text-amber-300 border border-amber-500/30"
            }`}
          >
            {status}
          </span>
        </div>
        <p className="text-xs text-slate-400 mt-0.5">
          {status === "healthy"
            ? `Response latency: ${latency} ms · Endpoint: ${api.getBaseUrl()}/health`
            : status === "offline"
            ? `Unable to reach ${api.getBaseUrl()}. Ensure FastAPI server is running.`
            : "Connecting to API..."}
        </p>
      </div>
    </div>
  );
}
