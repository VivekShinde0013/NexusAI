"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FileText,
  MessageSquare,
  Activity,
  ExternalLink,
  Home,
  Menu,
  X,
  Sparkles,
} from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";

export function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const docsUrl = `${api.getBaseUrl()}/docs`;

  const navItems = [
    { name: "Home", href: "/", icon: Home },
    { name: "Workspace", href: "/workspace", icon: LayoutDashboard },
    { name: "Documents", href: "/documents", icon: FileText },
    { name: "AI Chat", href: "/chat", icon: MessageSquare, badge: "RAG" },
    { name: "System Status", href: "/status", icon: Activity },
  ];

  return (
    <>
      {/* Mobile Menu Toggle Button */}
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="lg:hidden fixed bottom-5 right-5 z-50 p-3 rounded-full bg-blue-600 text-white shadow-xl shadow-blue-500/30 border border-blue-400/30 hover:bg-blue-500 transition-all"
        aria-label="Toggle navigation menu"
      >
        {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Overlay for mobile */}
      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          className="lg:hidden fixed inset-0 z-40 bg-slate-950/80 backdrop-blur-sm"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-16 bottom-0 left-0 z-40 w-64 border-r border-slate-800/80 bg-slate-950/90 backdrop-blur-xl transition-transform duration-300 lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        <div className="flex flex-col h-full justify-between p-4">
          <div className="space-y-6">
            <div>
              <p className="px-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
                Navigation
              </p>
              <nav className="space-y-1">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = pathname === item.href;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMobileOpen(false)}
                      className={`flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                        isActive
                          ? "bg-gradient-to-r from-blue-600/20 to-indigo-600/20 text-blue-400 border border-blue-500/30 shadow-md shadow-blue-500/10 font-semibold"
                          : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <Icon className={`w-4 h-4 ${isActive ? "text-blue-400" : "text-slate-400"}`} />
                        <span>{item.name}</span>
                      </div>
                      {item.badge && (
                        <span className="px-1.5 py-0.5 text-[10px] font-bold rounded-md bg-blue-500/20 text-blue-300 border border-blue-500/30">
                          {item.badge}
                        </span>
                      )}
                    </Link>
                  );
                })}
              </nav>
            </div>

            <div>
              <p className="px-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
                Developer Resources
              </p>
              <a
                href={docsUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 transition-all border border-transparent hover:border-slate-800"
              >
                <div className="flex items-center gap-3">
                  <ExternalLink className="w-4 h-4 text-indigo-400" />
                  <span>FastAPI Swagger UI</span>
                </div>
              </a>
            </div>
          </div>

          {/* Quick RAG Badge Card */}
          <div className="p-3.5 rounded-xl bg-gradient-to-b from-slate-900/90 to-slate-900/40 border border-slate-800/80">
            <div className="flex items-center gap-2 mb-1.5 text-blue-400 font-semibold text-xs">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Diversity RAG Active</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Subquery decomposition & FAISS top-K diversity filtering enabled.
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}
