"use client";

import { useState } from "react";
import {
  FileText,
  Trash2,
  Eye,
  RefreshCw,
  Search,
  AlertTriangle,
  Loader2,
  Calendar,
  Layers,
  HardDrive,
  FileCode,
  Sparkles,
} from "lucide-react";
import { DocumentMetadata } from "@/types/api";
import { formatBytes, formatDate } from "@/lib/utils";
import { api } from "@/lib/api";

interface DocumentTableProps {
  documents: DocumentMetadata[];
  loading: boolean;
  onRefresh: () => void;
}

export function DocumentTable({
  documents,
  loading,
  onRefresh,
}: DocumentTableProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedDoc, setSelectedDoc] = useState<DocumentMetadata | null>(null);
  const [docToDelete, setDocToDelete] = useState<DocumentMetadata | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const filteredDocs = documents.filter((doc) => {
    const term = searchTerm.toLowerCase();
    return (
      doc.original_filename.toLowerCase().includes(term) ||
      doc.file_type.toLowerCase().includes(term) ||
      doc.document_id.toLowerCase().includes(term)
    );
  });

  const handleDelete = async () => {
    if (!docToDelete) return;

    setDeletingId(docToDelete.document_id);
    setDeleteError(null);

    try {
      await api.deleteDocument(docToDelete.document_id);
      setDocToDelete(null);
      onRefresh();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to delete document.";
      setDeleteError(msg);
    } finally {
      setDeletingId(null);
    }
  };

  const getTypeBadgeColor = (fileType: string) => {
    switch (fileType.toLowerCase()) {
      case "pdf":
        return "bg-rose-500/10 text-rose-400 border-rose-500/20";
      case "docx":
        return "bg-blue-500/10 text-blue-400 border-blue-500/20";
      case "txt":
        return "bg-slate-500/10 text-slate-300 border-slate-500/20";
      case "csv":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
      default:
        return "bg-indigo-500/10 text-indigo-400 border-indigo-500/20";
    }
  };

  return (
    <div className="space-y-4">
      {/* Controls Header */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Filter documents..."
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-900/80 border border-slate-800 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
          <button
            onClick={onRefresh}
            disabled={loading}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-300 hover:text-white transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-blue-400" : ""}`} />
            <span>Refresh List</span>
          </button>
        </div>
      </div>

      {/* Main Table */}
      <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 overflow-hidden shadow-xl shadow-black/20">
        {loading && documents.length === 0 ? (
          <div className="p-12 text-center flex flex-col items-center justify-center gap-3">
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
            <p className="text-sm text-slate-400">Loading ingested documents...</p>
          </div>
        ) : filteredDocs.length === 0 ? (
          <div className="p-12 text-center space-y-3">
            <div className="w-12 h-12 rounded-2xl bg-slate-800/50 border border-slate-700/50 flex items-center justify-center mx-auto text-slate-400">
              <FileText className="w-6 h-6" />
            </div>
            <p className="text-sm font-semibold text-slate-300">
              {searchTerm ? "No matching documents found" : "No documents ingested yet"}
            </p>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              {searchTerm
                ? "Try adjusting your search query."
                : "Upload PDF, DOCX, TXT, or CSV files above to begin indexing."}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-950/60 border-b border-slate-800/80 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                <tr>
                  <th className="px-5 py-3.5">Document Name</th>
                  <th className="px-4 py-3.5">Type</th>
                  <th className="px-4 py-3.5">Size</th>
                  <th className="px-4 py-3.5">Chunks</th>
                  <th className="px-4 py-3.5">Status</th>
                  <th className="px-4 py-3.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {filteredDocs.map((doc) => (
                  <tr
                    key={doc.document_id}
                    className="hover:bg-slate-800/30 transition-colors group"
                  >
                    <td className="px-5 py-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50 text-blue-400 group-hover:border-blue-500/30 transition-colors">
                          <FileText className="w-4 h-4" />
                        </div>
                        <div>
                          <p className="font-semibold text-slate-100 line-clamp-1">
                            {doc.original_filename}
                          </p>
                          <p className="text-[10px] font-mono text-slate-500 mt-0.5">
                            ID: {doc.document_id}
                          </p>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <span
                        className={`text-[11px] font-extrabold uppercase px-2.5 py-1 rounded-md border ${getTypeBadgeColor(
                          doc.file_type
                        )}`}
                      >
                        {doc.file_type}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-xs text-slate-400">
                      {formatBytes(doc.file_size)}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-xs font-medium text-slate-300">
                      {doc.chunk_count !== undefined ? (
                        <span className="flex items-center gap-1.5 text-indigo-400">
                          <Layers className="w-3.5 h-3.5" />
                          {doc.chunk_count} chunks
                        </span>
                      ) : (
                        <span className="text-slate-500">Indexed</span>
                      )}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        {doc.status || "indexed"}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => setSelectedDoc(doc)}
                          className="p-1.5 rounded-lg bg-slate-800/50 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700/50 transition-colors"
                          title="View Details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setDocToDelete(doc)}
                          className="p-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 transition-colors"
                          title="Delete Document"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Document Metadata Inspector Modal */}
      {selectedDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
          <div className="w-full max-w-lg p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-100 text-base">
                    Document Details
                  </h3>
                  <p className="text-xs text-slate-400">Metadata & Vector Index Status</p>
                </div>
              </div>
              <button
                onClick={() => setSelectedDoc(null)}
                className="text-slate-400 hover:text-slate-200 text-sm font-semibold"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2">
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">Original Filename:</span>
                  <span className="font-semibold text-slate-200">{selectedDoc.original_filename}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">Document UUID:</span>
                  <span className="font-mono text-blue-400 text-[11px]">{selectedDoc.document_id}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">Stored Filename:</span>
                  <span className="font-mono text-slate-300 text-[11px]">{selectedDoc.stored_filename}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">MIME Type:</span>
                  <span className="text-slate-300">{selectedDoc.mime_type}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">File Size:</span>
                  <span className="text-slate-200 font-semibold">{formatBytes(selectedDoc.file_size)}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">Uploaded Date:</span>
                  <span className="text-slate-300">{formatDate(selectedDoc.uploaded_at)}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">FAISS Index Status:</span>
                  <span className="text-emerald-400 font-bold uppercase">{selectedDoc.status || "indexed"}</span>
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setSelectedDoc(null)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {docToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/85 backdrop-blur-md">
          <div className="w-full max-w-md p-6 rounded-2xl bg-slate-900 border border-rose-500/30 shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-bold text-slate-100 text-base">
                  Confirm Delete Document
                </h3>
                <p className="text-xs text-rose-400/90 font-medium">
                  This action cannot be undone.
                </p>
              </div>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Are you sure you want to delete <span className="font-bold text-white">{docToDelete.original_filename}</span>? This will remove the raw file, document metadata, and all associated vectors from the FAISS index.
            </p>

            {deleteError && (
              <p className="p-3 rounded-lg bg-rose-500/10 text-rose-300 text-xs border border-rose-500/20">
                {deleteError}
              </p>
            )}

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setDocToDelete(null)}
                disabled={!!deletingId}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={!!deletingId}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-xs font-semibold text-white shadow-lg shadow-rose-600/30 transition-all disabled:opacity-50"
              >
                {deletingId ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Deleting...</span>
                  </>
                ) : (
                  <span>Confirm Delete</span>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
