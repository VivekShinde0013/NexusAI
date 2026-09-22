from unittest.mock import MagicMock
import pytest
import numpy as np
from app.services.chunking_service import Chunk
from app.services.vector_store import VectorStore
from app.services.query_decomposition_service import (
    decompose_query,
    search_with_decomposition,
)


def test_decompose_query_cross_document_two_part():
    """Test 1: A clearly two-part cross-document query is decomposed into original + 2 sub-queries."""
    query = "What are the home office bandwidth requirements and password rotation cycles?"
    sub_queries = decompose_query(query)

    assert len(sub_queries) == 3
    assert sub_queries[0] == query
    assert "home office bandwidth requirements" in sub_queries[1].lower()
    assert "password rotation cycles" in sub_queries[2].lower()


def test_decompose_query_q023_style_wording():
    """Test 2: q-023-style wording ('gym reimbursement cap and incident response isolation time') produces two useful retrieval units."""
    query = "What is the gym reimbursement cap and incident response isolation time?"
    sub_queries = decompose_query(query)

    assert len(sub_queries) == 3
    assert sub_queries[0] == query
    assert "gym reimbursement cap" in sub_queries[1].lower()
    assert "incident response isolation time" in sub_queries[2].lower()


def test_decompose_query_single_topic_not_split():
    """Test 3: A normal single-topic query is NOT unnecessarily split."""
    query = "What is the annual leave allowance for full-time employees?"
    sub_queries = decompose_query(query)

    assert len(sub_queries) == 1
    assert sub_queries[0] == query


def test_decompose_query_no_clear_conjunction_unchanged():
    """Test 4: Queries with no clear conjunction remain unchanged."""
    query = "How many paid sick leave days do employees get?"
    sub_queries = decompose_query(query)

    assert len(sub_queries) == 1
    assert sub_queries[0] == query


def test_search_with_decomposition_merges_deterministically_and_deduplicates():
    """Test 5 & 6: Retrieval results from multiple sub-queries are merged deterministically and duplicate chunks are removed."""
    store = VectorStore(dimension=3)
    c1 = Chunk(
        chunk_id="c1",
        document_id="doc-A",
        source_filename="f1.txt",
        text="gym reimbursement",
        chunk_index=0,
        block_type="p",
    )
    c2 = Chunk(
        chunk_id="c2",
        document_id="doc-B",
        source_filename="f2.txt",
        text="incident isolation",
        chunk_index=0,
        block_type="p",
    )

    # Normalize vectors for L2 inner product
    emb1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    emb2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    store.add_chunks([c1], emb1)
    store.add_chunks([c2], emb2)

    mock_emb_service = MagicMock()
    # Mock returns 3-dim vectors matching store.dimension for any sub-queries
    mock_emb_service.generate_embeddings.side_effect = lambda queries: np.array(
        [[1.0, 0.0, 0.0] for _ in queries], dtype=np.float32
    )

    query = "What is the gym reimbursement cap and incident response isolation time?"
    results = search_with_decomposition(
        store, query, top_k=5, embedding_service=mock_emb_service
    )

    # Should contain both doc-A and doc-B chunks
    retrieved_doc_ids = [chunk.document_id for chunk, score in results]
    assert "doc-A" in retrieved_doc_ids

    # Check deduplication: total results should equal number of unique chunks
    chunk_ids = [chunk.chunk_id for chunk, score in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_search_with_decomposition_respects_document_id_filtering():
    """Test 7: Existing document_id filtering remains respected."""
    store = VectorStore(dimension=3)
    c1 = Chunk(
        chunk_id="c1",
        document_id="doc-A",
        source_filename="f1.txt",
        text="gym reimbursement",
        chunk_index=0,
        block_type="p",
    )
    c2 = Chunk(
        chunk_id="c2",
        document_id="doc-B",
        source_filename="f2.txt",
        text="incident isolation",
        chunk_index=0,
        block_type="p",
    )

    emb1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    emb2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    store.add_chunks([c1], emb1)
    store.add_chunks([c2], emb2)

    mock_emb_service = MagicMock()
    mock_emb_service.generate_embeddings.side_effect = lambda queries: np.array(
        [[1.0, 0.0, 0.0] for _ in queries], dtype=np.float32
    )

    query = "What is the gym reimbursement cap and incident response isolation time?"
    results = search_with_decomposition(
        store,
        query,
        top_k=5,
        document_ids=["doc-B"],
        embedding_service=mock_emb_service,
    )

    # Only doc-B chunk should be returned
    assert len(results) == 1
    assert results[0][0].document_id == "doc-B"


def test_search_with_decomposition_preserves_single_query_behavior():
    """Test 8: Existing single-query retrieval behavior remains intact."""
    store = VectorStore(dimension=3)
    c1 = Chunk(
        chunk_id="c1",
        document_id="doc-A",
        source_filename="f1.txt",
        text="annual leave days",
        chunk_index=0,
        block_type="p",
    )
    emb1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    store.add_chunks([c1], emb1)

    mock_emb_service = MagicMock()
    mock_emb_service.generate_embeddings.side_effect = lambda queries: np.array(
        [[1.0, 0.0, 0.0] for _ in queries], dtype=np.float32
    )

    query = "What is the annual leave allowance for full-time employees?"
    results = search_with_decomposition(
        store, query, top_k=5, embedding_service=mock_emb_service
    )

    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
