from datetime import datetime, timezone
import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.config import get_settings


class RAGTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    query_hash: str
    query_length: int
    query_preview: Optional[str] = None
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_usage_estimated: bool = True
    retrieved_count: int = 0
    top_score: float = 0.0
    relevance_threshold: float = 0.0
    retrieved_chunks: List[Dict[str, Any]] = []
    status: str = "success"
    error_details: Optional[str] = None


class TraceContext:

    def __init__(self, query: str):
        self.query = query
        self.start_time = time.perf_counter()

        # Precompute privacy-safe attributes
        self.query_length = len(query)
        self.query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()

        settings = get_settings()
        self.query_preview = (
            query[:100] if settings.LOG_QUERY_PREVIEWS else None
        )

        self.embedding_latency_ms = 0.0
        self.retrieval_latency_ms = 0.0
        self.llm_latency_ms = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.token_usage_estimated = True
        self.retrieved_count = 0
        self.top_score = 0.0
        self.relevance_threshold = settings.LLM_RELEVANCE_THRESHOLD
        self.retrieved_chunks = []
        self.status = "success"
        self.error_details = None

    def record_embedding(self, duration_ms: float):
        self.embedding_latency_ms = duration_ms

    def record_retrieval(
        self,
        duration_ms: float,
        chunks_and_scores: list,
        relevance_threshold: float,
    ):
        self.retrieval_latency_ms = duration_ms
        self.relevance_threshold = relevance_threshold
        self.retrieved_count = len(chunks_and_scores)
        if chunks_and_scores:
            self.top_score = float(max(score for _, score in chunks_and_scores))
        else:
            self.top_score = 0.0

        self.retrieved_chunks = []
        for chunk, score in chunks_and_scores:
            self.retrieved_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": float(score),
                    "document_id": chunk.document_id,
                    "page": chunk.page,
                    "block_type": chunk.block_type,
                }
            )

    def record_llm(
        self,
        duration_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated: bool,
    ):
        self.llm_latency_ms = duration_ms
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.token_usage_estimated = estimated

    def record_error(self, error_msg: str):
        self.status = "error"
        # Ensure we only log a safe summary (no keys, headers, stack traces)
        # Limit to the message, ignoring potential credentials/headers
        safe_msg = error_msg.split("\n")[0]
        if "sk-" in safe_msg or "Bearer" in safe_msg:
            safe_msg = "Internal API key/credential error."
        self.error_details = safe_msg

    def finalize_and_log(self) -> Dict[str, Any]:
        total_duration = (time.perf_counter() - self.start_time) * 1000.0

        trace = RAGTrace(
            query_hash=self.query_hash,
            query_length=self.query_length,
            query_preview=self.query_preview,
            embedding_latency_ms=self.embedding_latency_ms,
            retrieval_latency_ms=self.retrieval_latency_ms,
            llm_latency_ms=self.llm_latency_ms,
            total_latency_ms=total_duration,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            token_usage_estimated=self.token_usage_estimated,
            retrieved_count=self.retrieved_count,
            top_score=self.top_score,
            relevance_threshold=self.relevance_threshold,
            retrieved_chunks=self.retrieved_chunks,
            status=self.status,
            error_details=self.error_details,
        )

        settings = get_settings()
        log_dir = settings.get_logs_path()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "rag_traces.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

        return trace.model_dump()
