import {
  HealthResponse,
  DocumentMetadata,
  DocumentExtractionResponse,
  DocumentChunksResponse,
  DocumentIndexResponse,
  SearchResponse,
  ChatResponse,
  ChatRequest,
} from "@/types/api";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

class ApiClientError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error (${status}): ${detail}`);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP Error ${response.status} ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (typeof errorData.detail === "string") {
        errorMessage = errorData.detail;
      } else if (Array.isArray(errorData.detail)) {
        errorMessage = errorData.detail.map((e: { msg?: string }) => e.msg || JSON.stringify(e)).join("; ");
      } else if (errorData.message) {
        errorMessage = errorData.message;
      }
    } catch {
      // Failed to parse JSON error, use status text
    }
    throw new ApiClientError(response.status, errorMessage);
  }
  return response.json();
}

export const api = {
  getBaseUrl(): string {
    return API_BASE_URL;
  },

  async checkHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    return handleResponse<HealthResponse>(res);
  },

  async uploadDocument(file: File): Promise<DocumentMetadata> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    return handleResponse<DocumentMetadata>(res);
  },

  async extractDocument(documentId: string): Promise<DocumentExtractionResponse> {
    const res = await fetch(`${API_BASE_URL}/documents/${documentId}/extract`, {
      method: "POST",
    });
    return handleResponse<DocumentExtractionResponse>(res);
  },

  async chunkDocument(documentId: string): Promise<DocumentChunksResponse> {
    const res = await fetch(`${API_BASE_URL}/documents/${documentId}/chunks`, {
      method: "POST",
    });
    return handleResponse<DocumentChunksResponse>(res);
  },

  async indexDocument(documentId: string): Promise<DocumentIndexResponse> {
    const res = await fetch(`${API_BASE_URL}/documents/${documentId}/index`, {
      method: "POST",
    });
    return handleResponse<DocumentIndexResponse>(res);
  },

  async listDocuments(): Promise<DocumentMetadata[]> {
    const res = await fetch(`${API_BASE_URL}/documents`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    return handleResponse<DocumentMetadata[]>(res);
  },

  async getDocument(documentId: string): Promise<DocumentMetadata> {
    const res = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    return handleResponse<DocumentMetadata>(res);
  },

  async deleteDocument(documentId: string): Promise<{ message: string; document_id: string }> {
    const res = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
      method: "DELETE",
    });
    return handleResponse<{ message: string; document_id: string }>(res);
  },

  async search(query: string, topK = 5, documentIds?: string[]): Promise<SearchResponse> {
    const payload: { query: string; top_k: number; document_ids?: string[] } = {
      query,
      top_k: topK,
    };
    if (documentIds && documentIds.length > 0) {
      payload.document_ids = documentIds;
    }

    const res = await fetch(`${API_BASE_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return handleResponse<SearchResponse>(res);
  },

  async chatQuery(request: ChatRequest): Promise<ChatResponse> {
    const res = await fetch(`${API_BASE_URL}/chat/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    return handleResponse<ChatResponse>(res);
  },

  /**
   * Helper orchestrator that uploads a file and runs extract -> chunk -> index pipeline automatically!
   */
  async processDocumentPipeline(
    file: File,
    onStatusChange?: (status: string) => void
  ): Promise<DocumentMetadata> {
    onStatusChange?.("Uploading file...");
    const metadata = await this.uploadDocument(file);

    try {
      onStatusChange?.("Extracting text...");
      await this.extractDocument(metadata.document_id);

      onStatusChange?.("Chunking content...");
      await this.chunkDocument(metadata.document_id);

      onStatusChange?.("Vectorizing & Indexing in FAISS...");
      await this.indexDocument(metadata.document_id);

      onStatusChange?.("Completed!");
      // Fetch latest metadata with updated status/chunks
      return await this.getDocument(metadata.document_id);
    } catch (err) {
      console.warn("Pipeline processing issue for doc:", metadata.document_id, err);
      return metadata;
    }
  },
};
