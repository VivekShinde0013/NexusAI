"use client";

import { useState, useEffect } from "react";
import { Navbar } from "@/components/Navbar";
import { Sidebar } from "@/components/Sidebar";
import { ChatInterface } from "@/components/ChatInterface";
import { api } from "@/lib/api";
import { DocumentMetadata } from "@/types/api";

export default function ChatPage() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);

  useEffect(() => {
    api
      .listDocuments()
      .then((data) => setDocuments(data))
      .catch((err) => console.warn("Failed to load documents for chat:", err));
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100">
      <Navbar />

      <div className="flex flex-1">
        <Sidebar />

        <main className="flex-1 lg:pl-64 p-4 sm:p-6 lg:p-8 flex flex-col justify-center">
          <ChatInterface documents={documents} />
        </main>
      </div>
    </div>
  );
}
