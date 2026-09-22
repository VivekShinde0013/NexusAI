from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.services.document_extractor import ExtractedDocument


class Chunk(BaseModel):
    chunk_id: str  # Format: {document_id}-c{global_chunk_index}
    document_id: str
    source_filename: str
    text: str
    chunk_index: int  # 0-based global index within document
    page: Optional[int] = None  # 1-based page number for PDF
    block_type: str  # "page", "paragraph", "table", "document", "row"
    metadata: Dict[str, Any] = Field(default_factory=dict)


def chunk_document(
    doc: ExtractedDocument, chunk_size: int, chunk_overlap: int
) -> List[Chunk]:
    """Chunks an ExtractedDocument's blocks into retrieval-ready segments.

    Enforces limits, respects word boundaries, and ensures forward progress.
    """
    if chunk_size < 1 or chunk_size > 10000:
        raise ValueError("chunk_size must be between 1 and 10,000 characters.")
    if chunk_overlap < 0 or chunk_overlap > 2000:
        raise ValueError("chunk_overlap must be between 0 and 2,000 characters.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    chunks: List[Chunk] = []
    global_chunk_index = 0

    for block in doc.blocks:
        text = block.text
        if not text or not text.strip():
            continue

        text_len = len(text)
        start = 0

        while start < text_len:
            end = start + chunk_size

            # If remaining text fits, make one last chunk for this block
            if end >= text_len:
                chunk_text = text[start:].strip()
                if chunk_text:
                    page_num = block.index if doc.file_type == "pdf" else None
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.document_id}-c{global_chunk_index}",
                            document_id=doc.document_id,
                            source_filename=doc.source_filename,
                            text=chunk_text,
                            chunk_index=global_chunk_index,
                            page=page_num,
                            block_type=block.type,
                            metadata={"block_index": block.index},
                        )
                    )
                    global_chunk_index += 1
                break

            # Word boundary search: search backwards from 'end' to 'start + chunk_overlap'
            split_pos = -1
            # To ensure forward progress, split_pos must be strictly greater than start + chunk_overlap.
            # If search range is invalid (i.e. end <= start + chunk_overlap), we fall back directly.
            if end > start + chunk_overlap:
                for i in range(end, start + chunk_overlap, -1):
                    if text[i] in {" ", "\n", "\t", "\r"}:
                        split_pos = i
                        break

            if split_pos != -1:
                # Found whitespace boundary
                chunk_text = text[start:split_pos].strip()
                next_start = split_pos - chunk_overlap
            else:
                # Fallback to hard character boundary
                chunk_text = text[start:end].strip()
                next_start = end - chunk_overlap

            # Safe check: if next_start is not making progress, force it
            if next_start <= start:
                next_start = start + chunk_size - chunk_overlap

            if chunk_text:
                page_num = block.index if doc.file_type == "pdf" else None
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}-c{global_chunk_index}",
                        document_id=doc.document_id,
                        source_filename=doc.source_filename,
                        text=chunk_text,
                        chunk_index=global_chunk_index,
                        page=page_num,
                        block_type=block.type,
                        metadata={"block_index": block.index},
                    )
                )
                global_chunk_index += 1

            start = next_start

    return chunks
