import uuid
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.config import Settings, get_settings
from app.services.document_service import (
    generate_safe_filename,
    save_file,
    validate_file_extension,
    validate_mime_type,
)

router = APIRouter()
search_router = APIRouter()



@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="Filename is missing or empty"
        )

    # 1. Validate file extension
    ext = validate_file_extension(file.filename)

    # 2. Validate MIME type
    validate_mime_type(file.content_type, ext)

    # 3. Read content and validate size/emptiness
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the limit of {settings.MAX_UPLOAD_SIZE} bytes.",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 4. Generate IDs
    document_id = str(uuid.uuid4())
    stored_filename = generate_safe_filename(ext)

    # 5. Save the file
    upload_path = settings.get_upload_path()
    save_file(file_content, upload_path, stored_filename)

    # 6. Save metadata JSON using metadata_service
    from app.services import metadata_service

    try:
        metadata = metadata_service.create_metadata(
            document_id=document_id,
            original_filename=file.filename,
            stored_filename=stored_filename,
            file_type=ext.lstrip("."),
            file_size=file_size,
            mime_type=file.content_type,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create document metadata: {str(e)}"
        )

    # 7. Return JSON response
    return metadata



@router.post("/{document_id}/extract")
async def extract_document_text(
    document_id: str,
    settings: Settings = Depends(get_settings),
):
    # 1. Validate UUID format
    try:
        uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid document ID format. Must be a UUID."
        )

    # 2. Locate sidecar metadata JSON
    upload_path = settings.get_upload_path()
    metadata_path = upload_path / f"{document_id}.json"
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404, detail="Document metadata not found."
        )

    # 3. Read metadata
    import json

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Corrupted metadata file on disk."
        )

    stored_filename = metadata.get("stored_filename")
    original_filename = metadata.get("original_filename")
    file_type = metadata.get("file_type")

    if not stored_filename or not file_type:
        raise HTTPException(
            status_code=500, detail="Metadata values are incomplete."
        )

    # 4. Resolve raw file path and verify existence
    raw_file_path = upload_path / stored_filename
    if not raw_file_path.exists():
        raise HTTPException(
            status_code=404, detail="Raw document file not found on disk."
        )

    # 5. Extract text
    from app.services.document_extractor import extract_document

    try:
        extracted_doc = extract_document(
            document_id=document_id,
            original_filename=original_filename,
            file_path=raw_file_path,
            file_type=file_type,
        )
        return extracted_doc
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Text extraction failed: {str(e)}"
        )


@router.post("/{document_id}/chunks")
async def chunk_document_endpoint(
    document_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    settings: Settings = Depends(get_settings),
):
    # 1. Validate UUID format
    try:
        uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid document ID format. Must be a UUID."
        )

    # 2. Resolve settings and overrides
    c_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    c_overlap = (
        chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    )

    # 3. Validate bounds
    if c_size < 1 or c_size > 10000:
        raise HTTPException(
            status_code=400,
            detail="chunk_size override must be between 1 and 10,000 characters.",
        )
    if c_overlap < 0 or c_overlap > 2000:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap override must be between 0 and 2,000 characters.",
        )
    if c_overlap >= c_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap override must be strictly less than chunk_size.",
        )

    # 4. Locate sidecar metadata JSON
    upload_path = settings.get_upload_path()
    metadata_path = upload_path / f"{document_id}.json"
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404, detail="Document metadata not found."
        )

    # 5. Read metadata
    import json

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Corrupted metadata file on disk."
        )

    stored_filename = metadata.get("stored_filename")
    original_filename = metadata.get("original_filename")
    file_type = metadata.get("file_type")

    if not stored_filename or not file_type:
        raise HTTPException(
            status_code=500, detail="Metadata values are incomplete."
        )

    # 6. Resolve raw file path and verify existence
    raw_file_path = upload_path / stored_filename
    if not raw_file_path.exists():
        raise HTTPException(
            status_code=404, detail="Raw document file not found on disk."
        )

    # 7. Extract document text
    from app.services.document_extractor import extract_document

    try:
        extracted_doc = extract_document(
            document_id=document_id,
            original_filename=original_filename,
            file_path=raw_file_path,
            file_type=file_type,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Text extraction failed: {str(e)}"
        )

    # 8. Generate chunks
    from app.services.chunking_service import chunk_document

    try:
        chunks = chunk_document(extracted_doc, c_size, c_overlap)
        return chunks
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Document chunking failed: {str(e)}"
        )


@router.post("/{document_id}/index")
async def index_document_endpoint(
    document_id: str,
    settings: Settings = Depends(get_settings),
):
    # 1. Validate UUID format
    try:
        uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid document ID format. Must be a UUID."
        )

    # 2. Locate sidecar metadata JSON
    upload_path = settings.get_upload_path()
    metadata_path = upload_path / f"{document_id}.json"
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404, detail="Document metadata not found."
        )

    # 3. Read metadata
    import json

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Corrupted metadata file on disk."
        )

    stored_filename = metadata.get("stored_filename")
    original_filename = metadata.get("original_filename")
    file_type = metadata.get("file_type")

    if not stored_filename or not file_type:
        raise HTTPException(
            status_code=500, detail="Metadata values are incomplete."
        )

    # 4. Resolve raw file path and verify existence
    raw_file_path = upload_path / stored_filename
    if not raw_file_path.exists():
        raise HTTPException(
            status_code=404, detail="Raw document file not found on disk."
        )

    # 5. Extract document text
    from app.services.document_extractor import extract_document

    try:
        extracted_doc = extract_document(
            document_id=document_id,
            original_filename=original_filename,
            file_path=raw_file_path,
            file_type=file_type,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Text extraction failed: {str(e)}"
        )

    # 6. Generate chunks
    from app.services.chunking_service import chunk_document

    try:
        chunks = chunk_document(
            extracted_doc, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Document chunking failed: {str(e)}"
        )

    # 7. Run indexing service
    from app.services.indexing_service import index_chunks
    from app.services.embedding_service import get_embedding_service

    try:
        embedding_service = get_embedding_service()
        vector_dimension = embedding_service.get_dimension()

        chunks_indexed = index_chunks(document_id, chunks)

        # Update metadata status to indexed
        from app.services import metadata_service
        try:
            metadata_service.update_status(document_id, "indexed")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to update document metadata: {str(e)}"
            )

        return {
            "document_id": document_id,
            "chunks_indexed": chunks_indexed,
            "embedding_model": settings.EMBEDDING_MODEL,
            "vector_dimension": vector_dimension,
            "status": "indexed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Indexing failed: {str(e)}"
        )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_ids: Optional[list[str]] = None
    include_trace: bool = False


@search_router.post("/search")
async def search_endpoint(
    request: SearchRequest,
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
                    detail=f"Document '{doc_id}' is not indexed. Only indexed documents can be searched."
                )

    # 2. Get services
    from app.services.embedding_service import get_embedding_service
    from app.services.vector_store import get_vector_store
    from app.services.tracing_service import TraceContext
    import time

    try:
        embedding_service = get_embedding_service()
        store = get_vector_store()
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

    # 4. Perform search
    start_search = time.perf_counter()
    try:
        search_results = store.search(query_vector, request.top_k, document_ids=unique_doc_ids)
        trace_ctx.record_retrieval(
            (time.perf_counter() - start_search) * 1000.0,
            search_results,
            settings.LLM_RELEVANCE_THRESHOLD
        )
    except ValueError as ve:
        trace_ctx.record_error(str(ve))
        trace_ctx.finalize_and_log()
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        trace_ctx.record_error(str(e))
        trace_ctx.finalize_and_log()
        raise HTTPException(
            status_code=500, detail=f"Vector store search failed: {str(e)}"
        )

    # 5. Format results
    response_results = []
    for chunk, score in search_results:
        response_results.append(
            {
                "chunk_id": chunk.chunk_id,
                "score": float(score),
                "document_id": chunk.document_id,
                "source_filename": chunk.source_filename,
                "page": chunk.page,
                "block_type": chunk.block_type,
                "text": chunk.text,
            }
        )

    # Log structured trace
    trace_data = trace_ctx.finalize_and_log()

    # Determine whether to expose trace
    if request.include_trace and settings.DEBUG:
        return {"results": response_results, "trace": trace_data}

    return response_results


@router.get("")
async def list_documents_endpoint():
    """Lists all document metadata records."""
    from app.services import metadata_service

    try:
        return metadata_service.list_all_metadata()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/{document_id}")
async def get_document_endpoint(document_id: str):
    """Retrieves document metadata by ID."""
    try:
        uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid document ID format. Must be a UUID."
        )

    from app.services import metadata_service

    metadata = metadata_service.get_metadata(document_id)
    if not metadata:
        raise HTTPException(
            status_code=404, detail="Document metadata not found."
        )
    return metadata


@router.delete("/{document_id}")
async def delete_document_endpoint(
    document_id: str,
    settings: Settings = Depends(get_settings),
):
    """Deletes the document files and clean up its vectors from the FAISS store."""
    try:
        uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid document ID format. Must be a UUID."
        )

    from app.services import metadata_service
    from app.services.vector_store import get_vector_store

    # 1. Fetch metadata before deletion
    metadata = metadata_service.get_metadata(document_id)
    if not metadata:
        raise HTTPException(
            status_code=404, detail="Document metadata not found."
        )

    # 2. Delete raw and sidecar files from disk
    try:
        metadata_service.delete_document_files(document_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document files: {str(e)}"
        )

    # 3. Clean up from vector store
    try:
        store = get_vector_store()
        store.remove_document(document_id)
        store.save(settings.get_vector_store_path())
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear document vectors: {str(e)}"
        )

    return {"document_id": document_id, "status": "deleted"}




