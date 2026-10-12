"use client";

import Link from "next/link";
import { HealthBadge } from "./HealthBadge";
import { Cpu, ExternalLink } from "lucide-react";
import { api } from "@/lib/api";

export function Navbar() {
  const docsUrl = `${api.getBaseUrl()}/docs`;

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-purple-600 p-0.5 shadow-lg shadow-blue-500/20 group-hover:scale-105 transition-transform duration-200">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Cpu className="w-5 h-5 text-blue-400" />
              </div>
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent tracking-tight">
                Nexus<span className="text-blue-500">AI</span>
              </span>
              <span className="text-[10px] font-medium text-slate-500 uppercase tracking-widest -mt-1">
                Enterprise RAG
              </span>
            </div>
          </Link>
        </div>

        <div className="flex items-center gap-4">
          <HealthBadge compact />
          <a
            href={docsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-900/80 hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors"
          >
            <span>Swagger API Docs</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>
    </header>
  );
}
