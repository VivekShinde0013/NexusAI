"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  Send,
  Bot,
  User,
  Sparkles,
  FileText,
  Filter,
  Loader2,
  ChevronDown,
  ChevronUp,
  Activity,
  Layers,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Clock,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import { DocumentMetadata, ChatSource, TraceData } from "@/types/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  confidence?: "high" | "medium" | "low";
  trace?: TraceData;
  timestamp: string;
}

const SAMPLE_PROMPTS = [
  "What is the gym reimbursement cap?",
  "What is the incident response isolation time?",
  "How many annual leave days do employees receive?",
  "What are the security isolation requirements?",
];

interface ChatInterfaceProps {
  documents: DocumentMetadata[];
}

export function ChatInterface({ documents }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState<string>("all");
  const [includeTrace, setIncludeTrace] = useState(true);
  const [expandedTraceId, setExpandedTraceId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (queryText?: string) => {
    const query = (queryText || inputQuery).trim();
    if (!query || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage]);
    if (!queryText) setInputQuery("");
    setIsLoading(true);

    try {
      const docIds = selectedDocId === "all" ? undefined : [selectedDocId];
      const response = await api.chatQuery({
        query,
        top_k: 5,
        document_ids: docIds,
        include_trace: includeTrace,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        confidence: response.confidence,
        trace: response.trace,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: unknown) {
      const errorMsg =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while generating RAG response.";

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `**Error**: ${errorMsg}`,
        confidence: "low",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getConfidenceBadge = (confidence?: "high" | "medium" | "low") => {
    if (!confidence) return null;
    switch (confidence) {
      case "high":
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-0.5 rounded-full border border-emerald-500/20">
            <ShieldCheck className="w-3 h-3" /> High Confidence
          </span>
        );
      case "medium":
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-400 bg-amber-500/10 px-2.5 py-0.5 rounded-full border border-amber-500/20">
            <HelpCircle className="w-3 h-3" /> Medium Confidence
          </span>
        );
      case "low":
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-rose-400 bg-rose-500/10 px-2.5 py-0.5 rounded-full border border-rose-500/20">
            <AlertCircle className="w-3 h-3" /> Low Confidence / Fallback
          </span>
        );
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] max-w-6xl mx-auto rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-2xl overflow-hidden">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-5 py-3.5 bg-slate-950/70 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h2 className="font-bold text-slate-100 text-sm flex items-center gap-2">
              NexusAI Assistant
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-blue-500/20 text-blue-300 border border-blue-500/30">
                FAISS RAG
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Ask questions across indexed documents with hallucination guardrails.
            </p>
          </div>
        </div>

        {/* Document Filter & Trace Toggles */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl text-xs">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={selectedDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
              className="bg-transparent text-slate-200 focus:outline-none cursor-pointer"
            >
              <option value="all" className="bg-slate-900 text-slate-200">
                All Documents ({documents.filter((d) => d.status === "indexed").length})
              </option>
              {documents.map((doc) => {
                const isIndexed = doc.status === "indexed";
                return (
                  <option
                    key={doc.document_id}
                    value={doc.document_id}
                    disabled={!isIndexed}
                    className={isIndexed ? "bg-slate-900 text-slate-200" : "bg-slate-900 text-slate-500 italic"}
                  >
                    {doc.original_filename} {!isIndexed ? "(Unindexed)" : ""}
                  </option>
                );
              })}
            </select>
          </div>

          <label className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={includeTrace}
              onChange={(e) => setIncludeTrace(e.target.checked)}
              className="rounded bg-slate-900 border-slate-700 text-blue-500 focus:ring-blue-500/40"
            />
            <span>Show Traces</span>
          </label>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6 max-w-lg mx-auto py-12">
            <div className="p-4 rounded-3xl bg-gradient-to-tr from-blue-500/10 via-indigo-500/10 to-purple-500/10 border border-blue-500/20 shadow-xl">
              <Sparkles className="w-10 h-10 text-blue-400" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-bold text-slate-100">
                Ask anything about your documents
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                NexusAI uses vector search, deterministic query decomposition, and LLM reasoning to extract structured facts with exact source citations.
              </p>
            </div>

            <div className="w-full space-y-2 pt-2">
              <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider text-left">
                Suggested Prompts
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-left">
                {SAMPLE_PROMPTS.map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(prompt)}
                    className="p-3 rounded-xl bg-slate-900/80 hover:bg-slate-800/90 border border-slate-800 hover:border-blue-500/30 text-xs text-slate-300 hover:text-white transition-all text-left group"
                  >
                    <span className="group-hover:text-blue-400 transition-colors">
                      {prompt}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3.5 max-w-4xl ${
                msg.role === "user" ? "ml-auto flex-row-reverse" : ""
              }`}
            >
              {/* Avatar */}
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 text-xs font-bold ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white shadow-md shadow-blue-500/20"
                    : "bg-indigo-600/20 text-indigo-400 border border-indigo-500/30"
                }`}
              >
                {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              {/* Message Content Bubble */}
              <div className={`space-y-3 max-w-2xl ${msg.role === "user" ? "text-right" : ""}`}>
                <div
                  className={`p-4 rounded-2xl text-sm leading-relaxed text-slate-200 ${
                    msg.role === "user"
                      ? "bg-blue-600/90 text-white rounded-tr-none shadow-md shadow-blue-500/10"
                      : "bg-slate-900/90 border border-slate-800 rounded-tl-none"
                  }`}
                >
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>

                  <div className="mt-2 flex items-center gap-2 justify-end text-[10px] text-slate-400/80">
                    <span>{msg.timestamp}</span>
                  </div>
                </div>

                {/* Sources & Citations (for Assistant) */}
                {msg.role === "assistant" && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      {getConfidenceBadge(msg.confidence)}
                    </div>

                    {msg.sources && msg.sources.length > 0 && (
                      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2">
                        <div className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-400">
                          <FileText className="w-3.5 h-3.5 text-blue-400" />
                          <span>Source Citations ({msg.sources.length})</span>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {msg.sources.map((src, idx) => (
                            <div
                              key={idx}
                              className="p-2 rounded-lg bg-slate-900/80 border border-slate-800 text-xs space-y-1"
                            >
                              <div className="flex items-center justify-between font-semibold text-slate-200 truncate">
                                <span className="truncate">{src.source_filename}</span>
                                <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                                  {typeof src.score === "number" ? `${(src.score * 100).toFixed(0)}%` : "N/A"}
                                </span>
                              </div>
                              <div className="flex items-center justify-between text-[10px] text-slate-400">
                                <span>Page {src.page || 1}</span>
                                <span className="font-mono text-slate-500">{src.chunk_id}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Observability Trace Accordion */}
                    {msg.trace && (() => {
                      const totalMs = msg.trace.total_latency_ms ?? msg.trace.total_duration_ms;
                      const embedMs = msg.trace.embedding_latency_ms ?? msg.trace.embedding_duration_ms;
                      const retrievalMs = msg.trace.retrieval_latency_ms ?? msg.trace.retrieval_duration_ms;
                      const llmMs = msg.trace.llm_latency_ms ?? msg.trace.llm_duration_ms;

                      return (
                        <div className="rounded-xl bg-slate-950/40 border border-slate-800/60 overflow-hidden text-xs">
                          <button
                            onClick={() =>
                              setExpandedTraceId(
                                expandedTraceId === msg.id ? null : msg.id
                              )
                            }
                            className="w-full flex items-center justify-between px-3 py-2 text-[11px] text-slate-400 hover:text-slate-200 hover:bg-slate-900/50 transition-colors"
                          >
                            <div className="flex items-center gap-2">
                              <Activity className="w-3.5 h-3.5 text-indigo-400" />
                              <span>
                                Execution Trace ({typeof totalMs === "number" ? `${totalMs.toFixed(1)} ms` : "N/A"})
                              </span>
                            </div>
                            {expandedTraceId === msg.id ? (
                              <ChevronUp className="w-3.5 h-3.5" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5" />
                            )}
                          </button>

                          {expandedTraceId === msg.id && (
                            <div className="p-3 border-t border-slate-800/60 bg-slate-950/80 space-y-2 text-slate-300 font-mono text-[11px]">
                              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                  <p className="text-[10px] text-slate-500 uppercase">Embedding</p>
                                  <p className="font-bold text-blue-400">
                                    {typeof embedMs === "number" ? `${embedMs.toFixed(1)} ms` : "N/A"}
                                  </p>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                  <p className="text-[10px] text-slate-500 uppercase">Retrieval</p>
                                  <p className="font-bold text-indigo-400">
                                    {typeof retrievalMs === "number" ? `${retrievalMs.toFixed(1)} ms` : "N/A"}
                                  </p>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                  <p className="text-[10px] text-slate-500 uppercase">LLM Generation</p>
                                  <p className="font-bold text-purple-400">
                                    {typeof llmMs === "number" ? `${llmMs.toFixed(1)} ms` : "N/A"}
                                  </p>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                  <p className="text-[10px] text-slate-500 uppercase">Total Tokens</p>
                                  <p className="font-bold text-emerald-400">
                                    {typeof msg.trace.total_tokens === "number" ? msg.trace.total_tokens : 0}
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {/* Loading Spinner Indicator */}
        {isLoading && (
          <div className="flex items-center gap-3 text-slate-400 text-xs">
            <div className="w-8 h-8 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 flex items-center justify-center">
              <Loader2 className="w-4 h-4 animate-spin" />
            </div>
            <div className="p-3.5 rounded-2xl rounded-tl-none bg-slate-900/80 border border-slate-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400 animate-ping" />
              <span>Vectorizing query & synthesizing RAG context...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Box */}
      <div className="p-4 bg-slate-950/80 border-t border-slate-800/80">
        <div className="relative flex items-center bg-slate-900/90 rounded-2xl border border-slate-800 focus-within:border-blue-500/50 focus-within:ring-1 focus-within:ring-blue-500/50 transition-all">
          <textarea
            ref={textareaRef}
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents... (Press Enter to send)"
            disabled={isLoading}
            rows={2}
            className="w-full pl-4 pr-14 py-3 bg-transparent text-sm text-slate-200 placeholder-slate-500 resize-none focus:outline-none disabled:opacity-50"
          />

          <button
            onClick={() => handleSend()}
            disabled={!inputQuery.trim() || isLoading}
            className="absolute right-3 p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:hover:bg-blue-600 shadow-md shadow-blue-600/30 transition-all"
            aria-label="Send query"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-500 mt-2 px-1">
          <span>Shift + Enter for new line</span>
          <span>Hallucination Relevance Guardrail Threshold: 0.40</span>
        </div>
      </div>
    </div>
  );
}
