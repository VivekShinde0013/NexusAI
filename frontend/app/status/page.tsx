"use client";

import { useState, useEffect } from "react";
import { Navbar } from "@/components/Navbar";
import { Sidebar } from "@/components/Sidebar";
import { HealthBadge } from "@/components/HealthBadge";
import { api } from "@/lib/api";
import {
  Activity,
  CheckCircle2,
  XCircle,
  RefreshCw,
  ExternalLink,
  Shield,
  Server,
  Zap,
  Lock,
} from "lucide-react";

export default function StatusPage() {
  const [loading, setLoading] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [lastChecked, setLastChecked] = useState<string>("");

  const checkHealth = async () => {
    setLoading(true);
    const start = performance.now();
    try {
      const data = await api.checkHealth();
      const end = performance.now();
      setLatency(Math.round(end - start));
      setIsHealthy(data.status === "healthy");
    } catch {
      setIsHealthy(false);
      setLatency(null);
    } finally {
      setLoading(false);
      setLastChecked(new Date().toLocaleTimeString());
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const endpoints = [
    { method: "GET", path: "/health", description: "Health & readiness check", auth: "Public" },
    { method: "POST", path: "/documents/upload", description: "Upload PDF/DOCX/TXT/CSV file", auth: "Public" },
    { method: "POST", path: "/documents/{id}/extract", description: "Extract document plain text", auth: "Public" },
    { method: "POST", path: "/documents/{id}/chunks", description: "Chunk document text into sliding windows", auth: "Public" },
    { method: "POST", path: "/documents/{id}/index", description: "Vectorize & index chunks into FAISS", auth: "Public" },
    { method: "GET", path: "/documents", description: "List all document metadata", auth: "Public" },
    { method: "DELETE", path: "/documents/{id}", description: "Delete document & remove vectors", auth: "Public" },
    { method: "POST", path: "/search", description: "Semantic vector search with metadata filter", auth: "Public" },
    { method: "POST", path: "/chat/query", description: "Full RAG query with subquery decomposition", auth: "Public" },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100">
      <Navbar />

      <div className="flex flex-1">
        <Sidebar />

        <main className="flex-1 lg:pl-64 p-4 sm:p-6 lg:p-8 space-y-8 max-w-7xl mx-auto w-full">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
            <div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
                <Activity className="w-7 h-7 text-emerald-400 animate-pulse" />
                System Status & Diagnostics
              </h1>
              <p className="text-xs sm:text-sm text-slate-400 mt-1">
                Real-time API connectivity, latency benchmarks, and FastAPI route directory.
              </p>
            </div>

            <button
              onClick={checkHealth}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-300 hover:text-white transition-all disabled:opacity-50 w-fit"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-blue-400" : ""}`} />
              <span>Re-check Connectivity</span>
            </button>
          </div>

          {/* Health Badge Card */}
          <HealthBadge />

          {/* Health Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Backend Status
              </span>
              <div className="flex items-center gap-2 pt-1">
                {isHealthy ? (
                  <>
                    <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                    <span className="text-xl font-extrabold text-emerald-400">Operational</span>
                  </>
                ) : (
                  <>
                    <XCircle className="w-6 h-6 text-rose-400" />
                    <span className="text-xl font-extrabold text-rose-400">Offline / Error</span>
                  </>
                )}
              </div>
              <p className="text-xs text-slate-500">Last checked: {lastChecked || "N/A"}</p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Response Ping Latency
              </span>
              <div className="flex items-baseline gap-2 pt-1">
                <span className="text-2xl font-extrabold text-blue-400">
                  {latency ? `${latency} ms` : "N/A"}
                </span>
                <span className="text-xs text-emerald-400 font-semibold">Healthy</span>
              </div>
              <p className="text-xs text-slate-500">FastAPI Uvicorn Direct Ping</p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                API Base URL
              </span>
              <p className="text-sm font-mono font-semibold text-slate-200 truncate pt-1">
                {api.getBaseUrl()}
              </p>
              <p className="text-xs text-slate-500">Configured via NEXT_PUBLIC_API_BASE_URL</p>
            </div>
          </div>

          {/* Endpoints Table */}
          <div className="space-y-4 pt-2">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                <Server className="w-5 h-5 text-indigo-400" />
                Backend Endpoint Directory ({endpoints.length})
              </h2>
              <a
                href={`${api.getBaseUrl()}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 font-semibold"
              >
                <span>Interactive Swagger UI</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>

            <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 overflow-hidden shadow-xl">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-950/60 border-b border-slate-800/80 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    <tr>
                      <th className="px-5 py-3.5">Method</th>
                      <th className="px-5 py-3.5">Endpoint Path</th>
                      <th className="px-5 py-3.5">Description</th>
                      <th className="px-5 py-3.5">Auth</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {endpoints.map((ep, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-5 py-3.5 font-mono text-xs">
                          <span
                            className={`font-bold px-2 py-0.5 rounded ${
                              ep.method === "GET"
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                : ep.method === "POST"
                                ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                                : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                            }`}
                          >
                            {ep.method}
                          </span>
                        </td>
                        <td className="px-5 py-3.5 font-mono text-xs font-semibold text-slate-100">
                          {ep.path}
                        </td>
                        <td className="px-5 py-3.5 text-xs text-slate-400">{ep.description}</td>
                        <td className="px-5 py-3.5 text-xs text-slate-500">{ep.auth}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
