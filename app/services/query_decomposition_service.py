import re
from typing import List, Tuple, Optional, Dict
import numpy as np
from app.services.chunking_service import Chunk
from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService, get_embedding_service


def decompose_query(query: str) -> List[str]:
    """Decomposes a text query into independent sub-queries if multiple clauses/topics are detected.
    Always returns the original query as the first item in the list.
    """
    if not query or not query.strip():
        return [query]

    cleaned_query = query.strip()

    # 1. Check for multiple question mark sentences (e.g. "What is X? What is Y?")
    if cleaned_query.count("?") > 1:
        raw_parts = [p.strip() for p in cleaned_query.split("?") if p.strip()]
        if len(raw_parts) > 1:
            sub_queries = [cleaned_query]
            for part in raw_parts:
                sub_q = part + ("?" if not part.endswith("?") else "")
                if sub_q not in sub_queries:
                    sub_queries.append(sub_q)
            return sub_queries

    # 2. Check for conjunction splits: " and ", ", and ", "; ", " as well as ", " relate to ", " compared to "
    split_patterns = [
        (r"\b,?\s+and\s+", " and "),
        (r"\b;\s*", "; "),
        (r"\b\s+as well as\s+", " as well as "),
        (r"\b\s+relate to\s+", " relate to "),
        (r"\b\s+compared to\s+", " compared to "),
    ]

    matched_pattern = None
    for pattern_regex, label in split_patterns:
        match = re.search(pattern_regex, cleaned_query, flags=re.IGNORECASE)
        if match:
            matched_pattern = match
            break

    if matched_pattern:
        start_idx, end_idx = matched_pattern.span()
        clause1 = cleaned_query[:start_idx].strip()
        clause2 = cleaned_query[end_idx:].strip()

        # Clean trailing question marks or punctuation on clause1/clause2
        clause1_clean = clause1.rstrip("?").strip()
        clause2_clean = clause2.rstrip("?").strip()

        # Word count check: both clauses must contain at least 2 words
        words1 = clause1_clean.split()
        words2 = clause2_clean.split()

        if len(words1) >= 2 and len(words2) >= 2:
            question_words = (
                "what",
                "how",
                "where",
                "when",
                "who",
                "which",
                "can",
                "does",
                "is",
                "are",
                "do",
            )

            # Detect question prefix from clause1 (e.g., "What is the", "What are the", "How do")
            first_word = words1[0].lower() if words1 else ""
            prefix = ""
            if first_word in question_words:
                if len(words1) >= 2 and words1[1].lower() in (
                    "is",
                    "are",
                    "do",
                    "does",
                ):
                    prefix = f"{words1[0]} {words1[1]} "
                else:
                    prefix = f"{words1[0]} "

            if not clause2_clean.lower().startswith(question_words):
                clause2_formatted = f"{prefix}{clause2_clean}?"
            else:
                clause2_formatted = f"{clause2_clean}?"

            clause1_formatted = f"{clause1_clean}?"

            sub_queries = [cleaned_query]
            if clause1_formatted not in sub_queries:
                sub_queries.append(clause1_formatted)
            if clause2_formatted not in sub_queries:
                sub_queries.append(clause2_formatted)

            return sub_queries

    # Default: Return single query
    return [cleaned_query]


def search_with_decomposition(
    store: VectorStore,
    query: str,
    top_k: int,
    document_ids: Optional[List[str]] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> List[Tuple[Chunk, float]]:
    """Performs vector search with query decomposition and deterministic result merging.

    1. Decomposes query into sub-queries (including original query).
    2. Batch-generates embeddings for all sub-queries.
    3. Searches vector store for each sub-query.
    4. Merges & deduplicates candidate chunks, preserving highest similarity score per chunk.
    5. Applies VectorStore._select_diverse_chunks to select final top_k diverse results.
    """
    sub_queries = decompose_query(query)

    if embedding_service is None:
        embedding_service = get_embedding_service()

    if len(sub_queries) == 1:
        query_vector = embedding_service.generate_embeddings([sub_queries[0]])
        return store.search(query_vector, top_k=top_k, document_ids=document_ids)

    # Batch embedding generation for all sub-queries
    sub_vectors = embedding_service.generate_embeddings(sub_queries)

    # Collect and merge candidates from all sub-queries
    merged_candidates: Dict[str, Tuple[Chunk, float]] = {}

    for idx in range(len(sub_queries)):
        query_vec = sub_vectors[idx : idx + 1]
        results = store.search(query_vec, top_k=top_k, document_ids=document_ids)

        for chunk, score in results:
            cid = chunk.chunk_id
            if cid not in merged_candidates or score > merged_candidates[cid][1]:
                merged_candidates[cid] = (chunk, score)

    if not merged_candidates:
        return []

    # Sort merged candidates by score descending
    sorted_candidates = sorted(
        merged_candidates.values(), key=lambda x: x[1], reverse=True
    )

    # Apply document-diversity-aware selection
    return store._select_diverse_chunks(sorted_candidates, top_k=top_k)
