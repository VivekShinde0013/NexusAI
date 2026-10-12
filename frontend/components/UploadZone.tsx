"use client";

import { useState, useRef } from "react";
import { Upload, FileText, CheckCircle2, AlertCircle, Loader2, Sparkles, X } from "lucide-react";
import { api } from "@/lib/api";
import { formatBytes } from "@/lib/utils";
import { DocumentMetadata } from "@/types/api";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv"];
const ALLOWED_MIME_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
  "text/csv",
  "application/csv",
];

interface UploadZoneProps {
  onSuccess?: (doc: DocumentMetadata) => void;
  compact?: boolean;
}

export function UploadZone({ onSuccess, compact = false }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [stage, setStage] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [uploadedDoc, setUploadedDoc] = useState<DocumentMetadata | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return `Unsupported file format '${ext}'. Allowed formats: PDF, DOCX, TXT, CSV.`;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      return `File size (${formatBytes(file.size)}) exceeds the maximum allowed limit of 10 MB.`;
    }
    if (file.size === 0) {
      return "Uploaded file is empty.";
    }
    return null;
  };

  const handleFileSelect = async (file: File) => {
    setError(null);
    setUploadedDoc(null);

    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsUploading(true);
    try {
      const doc = await api.processDocumentPipeline(file, (stageText) => {
        setStage(stageText);
      });
      setUploadedDoc(doc);
      onSuccess?.(doc);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to process document upload.";
      setError(msg);
    } finally {
      setIsUploading(false);
      setStage("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="w-full space-y-4">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl transition-all duration-200 cursor-pointer overflow-hidden ${
          compact ? "p-6" : "p-8 sm:p-10"
        } ${
          isDragging
            ? "border-blue-500 bg-blue-500/10 shadow-[0_0_30px_rgba(59,130,246,0.2)]"
            : "border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/70"
        } ${isUploading ? "pointer-events-none opacity-80" : ""}`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFileSelect(e.target.files[0]);
            }
          }}
          accept={ALLOWED_EXTENSIONS.join(",")}
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center text-center space-y-3">
          <div
            className={`p-4 rounded-2xl bg-gradient-to-br from-blue-500/10 via-indigo-500/10 to-purple-500/10 border border-blue-500/20 shadow-lg ${
              isDragging ? "scale-110" : ""
            } transition-transform`}
          >
            {isUploading ? (
              <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
            ) : (
              <Upload className="w-8 h-8 text-blue-400" />
            )}
          </div>

          <div className="space-y-1">
            <p className="text-base font-semibold text-slate-200">
              {isUploading
                ? stage || "Processing document..."
                : "Drag & drop your document here"}
            </p>
            <p className="text-xs text-slate-400">
              or <span className="text-blue-400 underline font-medium">browse files</span> from your computer
            </p>
          </div>

          <div className="flex items-center gap-2 text-[11px] text-slate-500 pt-1">
            <span className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 font-mono">
              PDF, DOCX, TXT, CSV
            </span>
            <span>•</span>
            <span>Max size 10 MB</span>
          </div>

          {isUploading && (
            <div className="w-full max-w-xs pt-2">
              <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 animate-pulse w-3/4 rounded-full" />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="flex items-start justify-between gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold text-rose-200">Upload Failed</p>
              <p className="mt-0.5 text-rose-300/90">{error}</p>
            </div>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-rose-400 hover:text-rose-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Success Banner */}
      {uploadedDoc && (
        <div className="flex items-start justify-between gap-3 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs">
          <div className="flex items-start gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold text-emerald-200">
                Document Indexed Successfully!
              </p>
              <p className="mt-0.5 text-emerald-300/90">
                <span className="font-medium text-white">{uploadedDoc.original_filename}</span> ({formatBytes(uploadedDoc.file_size)}) was chunked and vectorized in FAISS.
              </p>
            </div>
          </div>
          <button
            onClick={() => setUploadedDoc(null)}
            className="text-emerald-400 hover:text-emerald-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
