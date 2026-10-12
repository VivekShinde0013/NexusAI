import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.config import Settings, get_settings
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.context_builder import build_context
from app.services.llm_service import get_llm_service

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    document_ids: Optional[list[str]] = None
    include_trace: bool = False


@router.post("/query")
async def chat_query_endpoint(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
):
    # 1. Input parameter validations
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400, detail="Query text cannot be empty."
        )

    if request.top_k < 1:
        raise HTTPException(
            status_code=400, detail="top_k must be at least 1."
        )

    # Validate document filters if supplied
    unique_doc_ids = None
    if request.document_ids is not None:
        # Validate UUID format
        for doc_id in request.document_ids:
            try:
                uuid.UUID(doc_id)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid document ID format in filter: '{doc_id}'."
                )

        # Deduplicate
        unique_doc_ids = list(dict.fromkeys(request.document_ids))

        # Validate existence & indexed status
        from app.services import metadata_service
        for doc_id in unique_doc_ids:
            meta = metadata_service.get_metadata(doc_id)
            if not meta:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document metadata not found for filter: '{doc_id}'."
                )
            if meta.get("status") != "indexed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Document '{doc_id}' is not indexed. Only indexed documents can be queried."
                )

    # 2. Get services
    from app.services.tracing_service import TraceContext
    import time

    try:
        embedding_service = get_embedding_service()
        store = get_vector_store()
        llm_service = get_llm_service()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Initialization failed: {str(e)}"
        )

    # Start tracing query
    trace_ctx = TraceContext(request.query)

    # 3. Generate query embedding
    start_embed = time.perf_counter()
    try:
        query_vector = embedding_service.generate_embeddings([request.query])
        trace_ctx.record_embedding((time.perf_counter() - start_embed) * 1000.0)
    except Exception as e:
        trace_ctx.record_error(str(e))
        trace_ctx.finalize_and_log()
        raise HTTPException(
            status_code=500, detail=f"Failed to generate query embedding: {str(e)}"
        )

    # 4. Determine effective threshold & perform search retrieval
    if unique_doc_ids:
        effective_threshold = settings.DOCUMENT_SCOPED_RELEVANCE_THRESHOLD
    else:
        effective_threshold = settings.LLM_RELEVANCE_THRESHOLD

    start_search = time.perf_counter()
    try:
        search_results = store.search(query_vector, request.top_k, document_ids=unique_doc_ids)
        trace_ctx.record_retrieval(
            (time.perf_counter() - start_search) * 1000.0,
            search_results,
            effective_threshold
        )
    except Exception as e:
        trace_ctx.record_error(str(e))
        trace_ctx.finalize_and_log()
        raise HTTPException(
            status_code=500, detail=f"Vector store search failed: {str(e)}"
        )

    # 5. Hallucination Protection & Filtering
    # Check if any chunk score is above effective relevance threshold
    highest_score = 0.0
    if search_results:
        highest_score = max(score for _, score in search_results)

    if highest_score < effective_threshold:
        # Skip calling LLM completely
        trace_ctx.record_llm(0.0, 0, 0, 0, True)
        trace_data = trace_ctx.finalize_and_log()
        response_payload = {
            "answer": "I could not find information about this in the provided documents.",
            "sources": [],
            "confidence": "low",
        }
        if request.include_trace and settings.DEBUG:
            response_payload["trace"] = trace_data
        return response_payload

    # 6. Source Deduplication (keep highest score per chunk_id)
    seen_chunks = {}
    for chunk, score in search_results:
        if score < effective_threshold:
            continue
        cid = chunk.chunk_id
        if cid not in seen_chunks or score > seen_chunks[cid][1]:
            seen_chunks[cid] = (chunk, score)

    filtered_results = list(seen_chunks.values())
    filtered_chunks = [item[0] for item in filtered_results]

    # 7. Context building
    context_text = build_context(filtered_chunks)

    # 8. Confidence classification
    if highest_score >= settings.LLM_HIGH_CONFIDENCE_THRESHOLD:
        confidence = "high"
    else:
        confidence = "medium"

    # 9. Invoke LLM Generation
    start_llm = time.perf_counter()
    try:
        answer = llm_service.generate_answer(request.query, context_text)
        duration_llm = (time.perf_counter() - start_llm) * 1000.0
        # Retrieve token counts from LLMService state
        token_info = getattr(
            llm_service,
            "last_token_info",
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated": True,
            },
        )
        trace_ctx.record_llm(
            duration_llm,
            token_info.get("prompt_tokens", 0),
            token_info.get("completion_tokens", 0),
            token_info.get("total_tokens", 0),
            token_info.get("estimated", True),
        )
    except ValueError as ve:
        trace_ctx.record_error(str(ve))
        trace_ctx.finalize_and_log()
        # Wrap client/connection/http errors cleanly as HTTP 502 Bad Gateway
        raise HTTPException(status_code=502, detail=str(ve))
    except Exception as e:
        trace_ctx.record_error(str(e))
        trace_ctx.finalize_and_log()
        raise HTTPException(
            status_code=500, detail="Internal RAG generation error."
        )

    # 10. Construct response
    sources_response = []
    for chunk, score in filtered_results:
        sources_response.append(
            {
                "source_filename": chunk.source_filename,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
            }
        )

    # Log trace to file
    trace_data = trace_ctx.finalize_and_log()

    response_payload = {
        "answer": answer,
        "sources": sources_response,
        "confidence": confidence,
    }
    if request.include_trace and settings.DEBUG:
        response_payload["trace"] = trace_data

    return response_payload
