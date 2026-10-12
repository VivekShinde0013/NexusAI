import Link from "next/link";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import {
  FileText,
  Search,
  Bot,
  Layers,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Zap,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
} from "lucide-react";
import { api } from "@/lib/api";

export default function LandingPage() {
  const docsUrl = `${api.getBaseUrl()}/docs`;

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100 selection:bg-blue-500/30 selection:text-blue-200">
      <Navbar />

      {/* Hero Section */}
      <section className="relative pt-16 pb-20 md:pt-24 md:pb-32 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Decorative Background Glows */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-gradient-to-tr from-blue-600/20 via-indigo-600/20 to-purple-600/20 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute top-1/3 left-1/4 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-6xl mx-auto text-center relative z-10 space-y-8">
          {/* Badge Pill */}
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900/90 border border-slate-800 text-xs font-semibold text-blue-400 shadow-xl shadow-blue-500/5">
            <Sparkles className="w-3.5 h-3.5 text-blue-400" />
            <span>FAISS + SentenceTransformers RAG Intelligence</span>
            <span className="text-slate-600">•</span>
            <span className="text-emerald-400">100% Retrieval Accuracy (Hit@5 = 1.00)</span>
          </div>

          {/* Main Hero Headline */}
          <div className="space-y-4 max-w-4xl mx-auto">
            <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white leading-[1.15]">
              Intelligent answers from your{" "}
              <span className="bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400 bg-clip-text text-transparent">
                enterprise documents.
              </span>
            </h1>
            <p className="text-base sm:text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
              NexusAI is a modular, provider-agnostic Retrieval-Augmented Generation (RAG) platform. Upload PDFs, Word documents, text, or CSVs and query them with deterministic subquery decomposition and hallucination guardrails.
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-2">
            <Link
              href="/workspace"
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 px-7 py-3.5 rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white font-bold text-sm shadow-xl shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-[1.02] transition-all"
            >
              <span>Open Workspace</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <a
              href={docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3.5 rounded-2xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white font-semibold text-sm transition-all"
            >
              <span>View API Docs</span>
              <ExternalLink className="w-4 h-4 text-slate-400" />
            </a>
          </div>

          {/* Abstract SVG / CSS Product Architecture Diagram */}
          <div className="pt-10 max-w-4xl mx-auto">
            <div className="p-6 sm:p-8 rounded-3xl bg-slate-900/70 border border-slate-800/80 shadow-2xl backdrop-blur-xl relative overflow-hidden group">
              <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6 text-xs text-slate-400">
                <span className="font-mono text-slate-300">NexusAI End-to-End RAG Architecture Pipeline</span>
                <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5" /> High Performance
                </span>
              </div>

              {/* Architecture Steps Flow */}
              <div className="grid grid-cols-2 md:grid-cols-7 gap-3 text-center items-center">
                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                  <FileText className="w-5 h-5 text-blue-400 mx-auto" />
                  <p className="text-[11px] font-bold text-slate-200">Documents</p>
                  <p className="text-[9px] text-slate-500">PDF/DOCX/TXT/CSV</p>
                </div>
                <div className="hidden md:block text-slate-600 text-xs font-mono">↓</div>

                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                  <Layers className="w-5 h-5 text-indigo-400 mx-auto" />
                  <p className="text-[11px] font-bold text-slate-200">Chunking</p>
                  <p className="text-[9px] text-slate-500">500 / 50 Overlap</p>
                </div>
                <div className="hidden md:block text-slate-600 text-xs font-mono">↓</div>

                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                  <Database className="w-5 h-5 text-purple-400 mx-auto" />
                  <p className="text-[11px] font-bold text-slate-200">FAISS Search</p>
                  <p className="text-[9px] text-slate-500">Dense Similarity</p>
                </div>
                <div className="hidden md:block text-slate-600 text-xs font-mono">↓</div>

                <div className="p-3 rounded-xl bg-gradient-to-tr from-blue-600/20 to-indigo-600/20 border border-blue-500/30 space-y-1 col-span-2 md:col-span-1">
                  <Bot className="w-5 h-5 text-emerald-400 mx-auto" />
                  <p className="text-[11px] font-bold text-emerald-300">RAG Answer</p>
                  <p className="text-[9px] text-emerald-400/80">+ Source Trace</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Cards Grid */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-slate-950/40 border-y border-slate-800/60">
        <div className="max-w-6xl mx-auto space-y-12">
          <div className="text-center space-y-2">
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white">
              Built for precision document intelligence
            </h2>
            <p className="text-sm text-slate-400 max-w-xl mx-auto">
              Every component is modular, benchmarked, and verified for production reliability.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700/80 transition-all space-y-3">
              <div className="p-3 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20 w-fit">
                <FileText className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-100 text-base">Multi-Format Ingestion</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Parse PDF, Word (.docx), plain text, and CSV data files with automated text extraction and metadata binding.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700/80 transition-all space-y-3">
              <div className="p-3 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 w-fit">
                <Search className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-100 text-base">Semantic FAISS Search</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                High-performance similarity retrieval using Sentence-Transformers (`all-MiniLM-L6-v2`) and FAISS CPU indexing.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700/80 transition-all space-y-3">
              <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20 w-fit">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-100 text-base">Hallucination Guardrails</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Relevance-score filtering (`LLM_RELEVANCE_THRESHOLD`) ensures the LLM is only called when context similarity is verified.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700/80 transition-all space-y-3">
              <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 w-fit">
                <Zap className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-100 text-base">Multi-Doc Decomposition</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Deterministic subquery extraction enables accurate cross-document reasoning across separate policies.
              </p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
