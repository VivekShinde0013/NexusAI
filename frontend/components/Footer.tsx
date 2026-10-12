import { Cpu } from "lucide-react";
import { api } from "@/lib/api";

export function Footer() {
  const docsUrl = `${api.getBaseUrl()}/docs`;

  return (
    <footer className="w-full border-t border-slate-800/80 bg-slate-950/60 py-8 px-4 sm:px-6 lg:px-8 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-slate-400 text-sm">
          <Cpu className="w-4 h-4 text-blue-500" />
          <span className="font-semibold text-slate-200">NexusAI</span>
          <span>— Production Enterprise Document Intelligence Platform</span>
        </div>

        <div className="flex items-center gap-6 text-xs text-slate-400">
          <a
            href={docsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-blue-400 transition-colors"
          >
            API Documentation
          </a>
          <span className="text-slate-700">•</span>
          <span>FastAPI + FAISS + SentenceTransformers</span>
          <span className="text-slate-700">•</span>
          <span>v0.1.0</span>
        </div>
      </div>
    </footer>
  );
}
