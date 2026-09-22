from typing import List
from app.services.chunking_service import Chunk


def build_context(chunks: List[Chunk]) -> str:
    """Formats retrieved chunks into a single structured context string for the LLM."""
    context_blocks = []
    for chunk in chunks:
        # Build block header indicating source info
        page_info = f"Page: {chunk.page}" if chunk.page is not None else "Page: N/A"
        header = f"[Chunk ID: {chunk.chunk_id}] Source: {chunk.source_filename} ({page_info} / Block Type: {chunk.block_type})"

        # Append formatted block
        context_blocks.append(f"{header}\nText: {chunk.text.strip()}")

    return "\n\n---\n\n".join(context_blocks)
