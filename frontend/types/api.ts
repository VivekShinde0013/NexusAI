export interface HealthResponse {
  status: string;
  service: string;
}

export interface DocumentMetadata {
  document_id: string;
  original_filename: string;
  stored_filename: string;
  file_type: string;
  file_size: number;
  mime_type: string;
  uploaded_at?: string;
  status?: string;
  chunk_count?: number;
}

export interface DocumentExtractionResponse {
  document_id: string;
  text: string;
  length: number;
}

export interface ChunkInfo {
  chunk_id: string;
  chunk_index: number;
  text: string;
  page?: number;
  start_char: number;
  end_char: number;
}

export interface DocumentChunksResponse {
  document_id: string;
  total_chunks: number;
  chunks: ChunkInfo[];
}

export interface DocumentIndexResponse {
  document_id: string;
  indexed_chunks: number;
  status: string;
}

export interface SearchResult {
  source_filename: string;
  page?: number;
  chunk_id: string;
  score: number;
  text?: string;
}

export interface SearchResponse {
  query: string;
  top_k: number;
  results: SearchResult[];
}

export interface ChatSource {
  source_filename: string;
  page?: number;
  chunk_id: string;
  score?: number;
}

export interface TraceData {
  query?: string;
  timestamp?: string;
  embedding_latency_ms?: number;
  retrieval_latency_ms?: number;
  llm_latency_ms?: number;
  total_latency_ms?: number;
  embedding_duration_ms?: number;
  retrieval_duration_ms?: number;
  llm_duration_ms?: number;
  total_duration_ms?: number;
  retrieved_count?: number;
  chunks_retrieved_count?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  token_usage_estimated?: boolean;
  estimated_tokens?: boolean;
  error?: string;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  confidence: "high" | "medium" | "low";
  trace?: TraceData;
}

export interface ChatRequest {
  query: string;
  top_k?: number;
  document_ids?: string[] | null;
  include_trace?: boolean;
}

export interface ApiError {
  detail: string | { msg: string }[];
}
