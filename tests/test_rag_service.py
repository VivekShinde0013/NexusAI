import tempfile
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings, Settings
from app.services.vector_store import get_vector_store
from app.services.llm_service import LLMService, get_llm_service
from app.services.chunking_service import Chunk

client = TestClient(app)

# --- Setup Mock SentenceTransformer ---


class MockSentenceTransformer:

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

    def encode(self, sentences, **kwargs):
        num_sentences = len(sentences)
        embeddings = []
        for text in sentences:
            seed = sum(ord(c) for c in text) % 1000
            np.random.seed(seed)
            vec = np.random.randn(384).astype(np.float32)
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
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_RELEVANCE_THRESHOLD", "0.4")
    monkeypatch.setenv("LLM_HIGH_CONFIDENCE_THRESHOLD", "0.6")

    get_settings.cache_clear()

    # Reset vector store to ensure clear index on each test run
    store = get_vector_store()
    store.clear()
    yield
    get_settings.cache_clear()


# --- Test Cases ---


def test_rag_query_success(temp_vector_dir):
    # Mock VectorStore.search to return a chunk with score 0.8
    mock_chunk = Chunk(
        chunk_id="doc-1-c0",
        document_id="doc-1",
        source_filename="cv.txt",
        text="Vivek Shinde knows Python, JavaScript, and Go.",
        chunk_index=0,
        page=None,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]

        # Query /chat/query
        chat_payload = {"query": "What programming languages does Vivek know?", "top_k": 2}
        chat_resp = client.post("/chat/query", json=chat_payload)
        assert chat_resp.status_code == 200
        data = chat_resp.json()

        # Assert response schema
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data

        # Verify answers and sources mapping
        assert "Python" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_filename"] == "cv.txt"
        assert data["sources"][0]["page"] is None
        assert "score" in data["sources"][0]
        assert data["confidence"] in ["high", "medium"]



def test_rag_query_validation():
    # Empty query validation
    resp_empty = client.post("/chat/query", json={"query": "   ", "top_k": 5})
    assert resp_empty.status_code == 400
    assert "cannot be empty" in resp_empty.json()["detail"]

    # top_k validation
    resp_topk = client.post("/chat/query", json={"query": "test", "top_k": 0})
    assert resp_topk.status_code == 400
    assert "must be at least 1" in resp_topk.json()["detail"]


def test_hallucination_protection():
    # Mock VectorStore.search to return a chunk with score < LLM_RELEVANCE_THRESHOLD (e.g. 0.3)
    mock_chunk = Chunk(
        chunk_id="doc-1-c0",
        document_id="doc-1",
        source_filename="doc.txt",
        text="This text is not matching query.",
        chunk_index=0,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search, \
         patch("app.services.llm_service.LLMService.generate_answer") as mock_generate:

        mock_search.return_value = [(mock_chunk, 0.3)]

        chat_payload = {"query": "Where is the gold?", "top_k": 2}
        resp = client.post("/chat/query", json=chat_payload)
        assert resp.status_code == 200
        data = resp.json()

        # LLMService should NOT be called
        mock_generate.assert_not_called()

        # Verify fallback response
        assert data["answer"] == "I could not find information about this in the provided documents."
        assert data["sources"] == []
        assert data["confidence"] == "low"


def test_source_deduplication():
    # Mock VectorStore.search to return duplicate chunk_ids with different scores
    mock_chunk = Chunk(
        chunk_id="doc-1-c0",
        document_id="doc-1",
        source_filename="doc.txt",
        text="Enterprise leaves policy details.",
        chunk_index=0,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        # Return duplicate chunks: score 0.5 and 0.8
        mock_search.return_value = [
            (mock_chunk, 0.5),
            (mock_chunk, 0.8),
        ]

        chat_payload = {"query": "leaves policy", "top_k": 5}
        resp = client.post("/chat/query", json=chat_payload)
        assert resp.status_code == 200
        data = resp.json()

        # Should be deduplicated to a single entry in sources
        assert len(data["sources"]) == 1
        # The preserved score must be the highest one (0.8)
        assert data["sources"][0]["score"] == 0.8
        assert data["confidence"] == "high"


def test_llm_api_error_handling():
    # Mock VectorStore.search to return a relevant chunk (score 0.8)
    mock_chunk = Chunk(
        chunk_id="doc-1-c0",
        document_id="doc-1",
        source_filename="doc.txt",
        text="Enterprise leaves policy details.",
        chunk_index=0,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search, \
         patch("app.services.llm_service.LLMService.generate_answer", side_effect=ValueError("LLM Request timed out")):

        mock_search.return_value = [(mock_chunk, 0.8)]

        chat_payload = {"query": "leaves policy", "top_k": 5}
        resp = client.post("/chat/query", json=chat_payload)

        # Expected status 502 Bad Gateway
        assert resp.status_code == 502
        # Error response should contain the clean ValueError message
        assert "timed out" in resp.json()["detail"].lower()


def test_confidence_boundaries():
    mock_chunk = Chunk(
        chunk_id="doc-1-c0",
        document_id="doc-1",
        source_filename="doc.txt",
        text="Standard context text.",
        chunk_index=0,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        # Case 1: High confidence score (0.7 >= 0.6)
        mock_search.return_value = [(mock_chunk, 0.7)]
        resp = client.post("/chat/query", json={"query": "test", "top_k": 5})
        assert resp.json()["confidence"] == "high"

        # Case 2: Medium confidence score (0.5 < 0.6 and >= 0.4)
        mock_search.return_value = [(mock_chunk, 0.5)]
        resp = client.post("/chat/query", json={"query": "test", "top_k": 5})
        assert resp.json()["confidence"] == "medium"


def test_mock_rag_does_not_invent_absent_facts():
    # Context contains C++, Java, C, Python, JavaScript but does not contain Go
    mock_chunk = Chunk(
        chunk_id="doc-1-c0",
        document_id="doc-1",
        source_filename="cv.txt",
        text="Vivek Shinde has experience in programming with C++, Java, C, Python, and JavaScript.",
        chunk_index=0,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]

        chat_payload = {"query": "What programming languages does Vivek know?", "top_k": 5}
        resp = client.post("/chat/query", json=chat_payload)
        assert resp.status_code == 200
        data = resp.json()

        # Verifies the mock generation dynamics: Go is absent from context, so must be absent from answer
        assert "Go" not in data["answer"]
        assert "Python" in data["answer"]
        assert "JavaScript" in data["answer"]
        assert "C++" in data["answer"]

