"use client";

import { useState, useEffect } from "react";
import { Navbar } from "@/components/Navbar";
import { Sidebar } from "@/components/Sidebar";
import { UploadZone } from "@/components/UploadZone";
import { DocumentTable } from "@/components/DocumentTable";
import { api } from "@/lib/api";
import { DocumentMetadata } from "@/types/api";
import { FileText, Plus } from "lucide-react";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const data = await api.listDocuments();
      setDocuments(data);
    } catch (err) {
      console.error("Failed to list documents:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

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
                <FileText className="w-7 h-7 text-blue-400" />
                Document Lifecycle Management
              </h1>
              <p className="text-xs sm:text-sm text-slate-400 mt-1">
                Upload, extract, chunk, index, inspect, and manage enterprise documents in FAISS.
              </p>
            </div>

            <button
              onClick={() => setShowUpload(!showUpload)}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs shadow-lg shadow-blue-500/20 transition-all w-fit"
            >
              <Plus className="w-4 h-4" />
              <span>{showUpload ? "Hide Upload Zone" : "Upload New File"}</span>
            </button>
          </div>

          {/* Upload Zone Component */}
          {(showUpload || documents.length === 0) && (
            <div className="space-y-3">
              <UploadZone onSuccess={fetchDocuments} />
            </div>
          )}

          {/* Document Table */}
          <div className="space-y-4">
            <h2 className="text-lg font-bold text-slate-100">
              All Ingested Documents ({documents.length})
            </h2>

            <DocumentTable
              documents={documents}
              loading={loading}
              onRefresh={fetchDocuments}
            />
          </div>
        </main>
      </div>
    </div>
  );
}
