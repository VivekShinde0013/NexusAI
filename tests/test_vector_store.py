import os
import tempfile
import json
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch
from app.config import get_settings, Settings
from app.services.chunking_service import Chunk
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store
from app.services.indexing_service import index_chunks

# --- Setup Mock SentenceTransformer ---


class MockSentenceTransformer:

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

    def encode(self, sentences, **kwargs):
        num_sentences = len(sentences)
        # Create deterministic pseudo-embeddings for testing
        embeddings = []
        for text in sentences:
            # Deterministic vector based on sum of ascii values
            seed = sum(ord(c) for c in text) % 1000
            np.random.seed(seed)
            vec = np.random.randn(384).astype(np.float32)
            # Normalize vector
            norm = np.linalg.norm(vec)
            vec = vec / (norm if norm > 0 else 1.0)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return 384


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    with patch(
        "app.services.embedding_service.SentenceTransformer",
        MockSentenceTransformer,
    ):
        yield


@pytest.fixture
def temp_vector_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def override_settings(temp_vector_dir, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(temp_vector_dir / "raw"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(temp_vector_dir / "vectors"))
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "10000000")

    get_settings.cache_clear()

    # Reset vector store to ensure clear index on each test run
    store = get_vector_store()
    store.clear()
    yield
    get_settings.cache_clear()




# --- Unit & Service Level Tests ---


def test_embedding_service_mock():
    service = EmbeddingService("all-MiniLM-L6-v2")
    assert service.get_dimension() == 384

    # Test single-text embedding shape and normalization
    embs = service.generate_embeddings(["test text"])
    assert embs.shape == (1, 384)
    # Check L2 normalized: norm should be extremely close to 1.0
    norm = np.linalg.norm(embs[0])
    assert pytest.approx(norm, abs=1e-5) == 1.0

    # Test batch embedding
    batch_embs = service.generate_embeddings(["text A", "text B", "text C"])
    assert batch_embs.shape == (3, 384)

    # Test empty list
    empty_embs = service.generate_embeddings([])
    assert empty_embs.shape == (0, 384)


def test_vector_store_persistence_and_reload(temp_vector_dir):
    store = VectorStore(dimension=384)

    chunk_a = Chunk(
        chunk_id="doc-A-c0",
        document_id="doc-A",
        source_filename="fileA.txt",
        text="Sample paragraph A.",
        chunk_index=0,
        block_type="paragraph",
    )
    chunk_b = Chunk(
        chunk_id="doc-B-c0",
        document_id="doc-B",
        source_filename="fileB.txt",
        text="Sample paragraph B.",
        chunk_index=0,
        block_type="paragraph",
    )

    embeddings = np.random.randn(2, 384).astype(np.float32)
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    store.add_chunks([chunk_a, chunk_b], embeddings)
    assert store.count() == 2

    # Save
    store.save(temp_vector_dir)

    # Load into a fresh vector store
    new_store = VectorStore(dimension=384)
    new_store.load(temp_vector_dir)
    assert new_store.count() == 2
    assert new_store.chunks[0].chunk_id == "doc-A-c0"
    assert new_store.chunks[1].chunk_id == "doc-B-c0"


def test_vector_store_corruption_check(temp_vector_dir):
    store = VectorStore(dimension=384)
    chunk = Chunk(
        chunk_id="doc-A-c0",
        document_id="doc-A",
        source_filename="fileA.txt",
        text="Text.",
        chunk_index=0,
        block_type="paragraph",
    )
    emb = np.random.randn(1, 384).astype(np.float32)
    store.add_chunks([chunk], emb)

    store.save(temp_vector_dir)

    # Corrupt metadata.json by appending an extra chunk manually
    metadata_path = temp_vector_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        meta_data = json.load(f)

    meta_data.append(meta_data[0])  # Duplicate the chunk in JSON
    with open(metadata_path, "w") as f:
        json.dump(meta_data, f)

    # Reload should fail due to size mismatch
    corrupted_store = VectorStore(dimension=384)
    with pytest.raises(ValueError, match="Vector store corruption"):
        corrupted_store.load(temp_vector_dir)


def test_vector_store_empty_handling():
    store = VectorStore(dimension=384)
    query_vector = np.random.randn(1, 384).astype(np.float32)

    # Search on empty store must return [] rather than raising FAISS error
    results = store.search(query_vector, top_k=5)
    assert results == []


def test_vector_store_restart_simulation_integration(temp_vector_dir):
    # 1. Create and persist vectors
    store = VectorStore(dimension=384)
    chunk = Chunk(
        chunk_id="doc-1-c0",
        document_id="doc-1",
        source_filename="doc.txt",
        text="FastAPI restart vector store simulation test.",
        chunk_index=0,
        block_type="document",
    )
    emb = np.random.randn(1, 384).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    store.add_chunks([chunk], emb)
    store.save(temp_vector_dir)

    # 2. Discard instance and create a fresh one
    del store

    # 3. Load from disk
    new_store = VectorStore(dimension=384)
    new_store.load(temp_vector_dir)

    # 4. Search and verify metadata alignment
    results = new_store.search(emb, top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "doc-1-c0"
    assert results[0][0].text == "FastAPI restart vector store simulation test."


def test_indexing_service_duplicate_rebuild_logic(temp_vector_dir):
    # Setup test settings to use temp vector store dir
    with patch(
        "app.services.indexing_service.get_settings"
    ) as mock_get_settings:
        mock_settings = Settings(
            VECTOR_STORE_DIR=str(temp_vector_dir),
            EMBEDDING_MODEL="all-MiniLM-L6-v2",
        )
        mock_get_settings.return_value = mock_settings

        store = get_vector_store()
        store.clear()

        # 1. Index document A
        chunk_a1 = Chunk(
            chunk_id="doc-A-c0",
            document_id="doc-A",
            source_filename="A.txt",
            text="Document A chunk 1 text content.",
            chunk_index=0,
            block_type="paragraph",
        )
        index_chunks("doc-A", [chunk_a1])

        # 2. Index document B
        chunk_b1 = Chunk(
            chunk_id="doc-B-c0",
            document_id="doc-B",
            source_filename="B.txt",
            text="Document B chunk 1 text content.",
            chunk_index=0,
            block_type="paragraph",
        )
        index_chunks("doc-B", [chunk_b1])

        assert store.count() == 2

        # 3. Re-index document A with updated content (2 chunks)
        chunk_a1_new = Chunk(
            chunk_id="doc-A-c0",
            document_id="doc-A",
            source_filename="A.txt",
            text="Updated document A chunk 1 text content.",
            chunk_index=0,
            block_type="paragraph",
        )
        chunk_a2_new = Chunk(
            chunk_id="doc-A-c1",
            document_id="doc-A",
            source_filename="A.txt",
            text="Updated document A chunk 2 text content.",
            chunk_index=1,
            block_type="paragraph",
        )

        index_chunks("doc-A", [chunk_a1_new, chunk_a2_new])

        # 4. Verify count matches (1 for Doc B + 2 for Doc A = 3)
        assert store.count() == 3
        assert len(store.chunks) == 3

        # 5. Verify document B vectors remain intact and document A is updated
        ids = [c.chunk_id for c in store.chunks]
        assert "doc-B-c0" in ids
        assert "doc-A-c0" in ids
        assert "doc-A-c1" in ids

        # Confirm A content is updated
        for chunk in store.chunks:
            if chunk.chunk_id == "doc-A-c0":
                assert "Updated" in chunk.text


# --- Real Embedding Integration Test (Optional) ---

try:
    from sentence_transformers import SentenceTransformer

    # Try loading real model with local files only
    SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    REAL_MODEL_AVAILABLE = True
except Exception:
    REAL_MODEL_AVAILABLE = False


@pytest.mark.skipif(
    not REAL_MODEL_AVAILABLE,
    reason="Real all-MiniLM-L6-v2 model is not cached locally.",
)
def test_real_embedding_model_integration():
    # If the model is available locally, run a test with the real model
    real_service = EmbeddingService("all-MiniLM-L6-v2")
    embs = real_service.generate_embeddings(["This is a test of real embeddings"])
    assert embs.shape == (1, 384)
    assert np.allclose(np.linalg.norm(embs[0]), 1.0, atol=1e-5)


# --- Document Diversity Retrieval Tests ---

def make_synthetic_embedding(score: float, dim: int = 384) -> np.ndarray:
    vec = np.zeros((1, dim), dtype=np.float32)
    vec[0, 0] = score
    vec[0, 1] = np.sqrt(max(0.0, 1.0 - score**2))
    return vec


def test_vector_store_diversity_cross_document_crowding():
    """Test A: Cross-document query where one document has many higher-scoring chunks.
    Both relevant documents survive into the final Top-K when candidates from both are available.
    """
    store = VectorStore(dimension=384)
    chunks = []
    embeddings = []
    for i in range(6):
        c = Chunk(
            chunk_id=f"doc-A-c{i}",
            document_id="doc-A",
            source_filename="A.txt",
            text=f"Text A {i}",
            chunk_index=i,
            block_type="paragraph"
        )
        chunks.append(c)
        embeddings.append(make_synthetic_embedding(0.95 - i * 0.01)[0])

    for j in range(2):
        c = Chunk(
            chunk_id=f"doc-B-c{j}",
            document_id="doc-B",
            source_filename="B.txt",
            text=f"Text B {j}",
            chunk_index=j,
            block_type="paragraph"
        )
        chunks.append(c)
        embeddings.append(make_synthetic_embedding(0.85 - j * 0.01)[0])

    store.add_chunks(chunks, np.array(embeddings, dtype=np.float32))

    query_vec = make_synthetic_embedding(1.0)
    results = store.search(query_vec, top_k=5)

    assert len(results) == 5
    retrieved_doc_ids = [c.document_id for c, _ in results]
    assert "doc-A" in retrieved_doc_ids
    assert "doc-B" in retrieved_doc_ids
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_store_diversity_single_document_query():
    """Test B: Query where only one document is relevant.
    Multiple useful chunks from that document should still be returned up to top_k.
    """
    store = VectorStore(dimension=384)
    chunks = []
    embeddings = []
    for i in range(6):
        c = Chunk(
            chunk_id=f"doc-A-c{i}",
            document_id="doc-A",
            source_filename="A.txt",
            text=f"Text A {i}",
            chunk_index=i,
            block_type="paragraph"
        )
        chunks.append(c)
        embeddings.append(make_synthetic_embedding(0.95 - i * 0.01)[0])

    store.add_chunks(chunks, np.array(embeddings, dtype=np.float32))

    query_vec = make_synthetic_embedding(1.0)
    results = store.search(query_vec, top_k=5)

    assert len(results) == 5
    for chunk, _ in results:
        assert chunk.document_id == "doc-A"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_store_diversity_fewer_chunks_than_top_k():
    """Test C: Query with fewer total chunks/documents than top_k.
    Retrieval should still return as many valid chunks as available.
    """
    store = VectorStore(dimension=384)
    chunks = [
        Chunk(chunk_id="doc-A-c0", document_id="doc-A", source_filename="A.txt", text="Text A", chunk_index=0, block_type="paragraph"),
        Chunk(chunk_id="doc-B-c0", document_id="doc-B", source_filename="B.txt", text="Text B", chunk_index=0, block_type="paragraph"),
    ]
    embs = np.array([make_synthetic_embedding(0.9)[0], make_synthetic_embedding(0.8)[0]], dtype=np.float32)
    store.add_chunks(chunks, embs)

    query_vec = make_synthetic_embedding(1.0)
    results = store.search(query_vec, top_k=5)

    assert len(results) == 2
    assert results[0][0].chunk_id == "doc-A-c0"
    assert results[1][0].chunk_id == "doc-B-c0"


def test_vector_store_diversity_with_document_filter():
    """Test D: Document filtering combined with diversity retrieval."""
    store = VectorStore(dimension=384)
    chunks = [
        Chunk(chunk_id="doc-A-c0", document_id="doc-A", source_filename="A.txt", text="Text A0", chunk_index=0, block_type="paragraph"),
        Chunk(chunk_id="doc-A-c1", document_id="doc-A", source_filename="A.txt", text="Text A1", chunk_index=1, block_type="paragraph"),
        Chunk(chunk_id="doc-B-c0", document_id="doc-B", source_filename="B.txt", text="Text B0", chunk_index=0, block_type="paragraph"),
    ]
    embs = np.array([
        make_synthetic_embedding(0.9)[0],
        make_synthetic_embedding(0.85)[0],
        make_synthetic_embedding(0.95)[0],
    ], dtype=np.float32)
    store.add_chunks(chunks, embs)

    query_vec = make_synthetic_embedding(1.0)
    results = store.search(query_vec, top_k=5, document_ids=["doc-A"])

    assert len(results) == 2
    assert all(c.document_id == "doc-A" for c, _ in results)


def test_vector_store_diversity_deterministic_ordering():
    """Test E: Diversity retrieval must be 100% deterministic."""
    store = VectorStore(dimension=384)
    chunks = []
    embeddings = []
    for i in range(10):
        doc_id = f"doc-{i % 3}"
        c = Chunk(
            chunk_id=f"c-{i}",
            document_id=doc_id,
            source_filename=f"{doc_id}.txt",
            text=f"Text {i}",
            chunk_index=i,
            block_type="paragraph"
        )
        chunks.append(c)
        embeddings.append(make_synthetic_embedding(0.9 - i * 0.02)[0])

    store.add_chunks(chunks, np.array(embeddings, dtype=np.float32))
    query_vec = make_synthetic_embedding(1.0)

    res1 = store.search(query_vec, top_k=5)
    res2 = store.search(query_vec, top_k=5)

    assert len(res1) == len(res2)
    for (c1, s1), (c2, s2) in zip(res1, res2):
        assert c1.chunk_id == c2.chunk_id
        assert s1 == s2

