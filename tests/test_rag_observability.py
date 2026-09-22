import os
import tempfile
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings, Settings
from app.services.vector_store import get_vector_store
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
    monkeypatch.setenv("LOGS_DIR", str(temp_vector_dir / "logs"))
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "10000000")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_RELEVANCE_THRESHOLD", "0.4")
    monkeypatch.setenv("LLM_HIGH_CONFIDENCE_THRESHOLD", "0.6")
    # Enable query previews in configuration to verify conditional preview logging
    monkeypatch.setenv("LOG_QUERY_PREVIEWS", "true")
    # Default tests run in debug mode
    monkeypatch.setenv("DEBUG", "true")

    get_settings.cache_clear()

    # Reset vector store to ensure clear index on each test run
    store = get_vector_store()
    store.clear()
    yield
    get_settings.cache_clear()


def _get_log_lines(temp_vector_dir):
    log_file = temp_vector_dir / "logs" / "rag_traces.jsonl"
    if not log_file.exists():
        return []
    with open(log_file, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


# --- TDD Test Cases ---


def test_search_tracing_enabled(temp_vector_dir):
    # Upload & index Doc A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Hello observational world of search.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    # Search with include_trace=True
    resp = client.post(
        "/search", json={"query": "world", "top_k": 5, "include_trace": True}
    )
    assert resp.status_code == 200
    data = resp.json()

    # Verify trace in API response
    assert "trace" in data
    trace = data["trace"]
    assert "trace_id" in trace
    assert trace["total_latency_ms"] >= 0
    assert trace["embedding_latency_ms"] >= 0
    assert trace["retrieval_latency_ms"] >= 0
    assert trace["retrieved_count"] == 1
    assert trace["top_score"] is not None

    # Verify trace is written to disk
    logs = _get_log_lines(temp_vector_dir)
    assert len(logs) == 1
    log_entry = logs[0]
    assert log_entry["trace_id"] == trace["trace_id"]
    assert log_entry["status"] == "success"


def test_chat_tracing_enabled(temp_vector_dir):
    # Upload & index Doc A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Unique text in Document A about leaves.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    mock_chunk = Chunk(
        chunk_id="docA-c0",
        document_id=doc_a_id,
        source_filename="docA.txt",
        text="Unique text in Document A about leaves.",
        chunk_index=0,
        page=None,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]

        resp = client.post(
            "/chat/query",
            json={"query": "leaves", "top_k": 5, "include_trace": True},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "trace" in data
        trace = data["trace"]
        assert trace["total_latency_ms"] >= 0
        assert trace["embedding_latency_ms"] >= 0
        assert trace["retrieval_latency_ms"] >= 0
        assert trace["llm_latency_ms"] >= 0
        assert trace["prompt_tokens"] > 0
        assert trace["completion_tokens"] > 0
        assert trace["total_tokens"] > 0
        assert trace["token_usage_estimated"] is True
        assert trace["retrieved_count"] == 1
        assert trace["top_score"] == 0.8
        assert trace["relevance_threshold"] == 0.4


def test_chat_tracing_disabled(temp_vector_dir):
    # Upload & index Doc A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Unique text in Document A about leaves.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    mock_chunk = Chunk(
        chunk_id="docA-c0",
        document_id=doc_a_id,
        source_filename="docA.txt",
        text="Unique text in Document A about leaves.",
        chunk_index=0,
        page=None,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]

        # include_trace=False
        resp = client.post(
            "/chat/query",
            json={"query": "leaves", "top_k": 5, "include_trace": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "trace" not in data

        # Check logs on disk to verify trace was still generated and logged
        logs = _get_log_lines(temp_vector_dir)
        assert len(logs) == 1
        assert logs[0]["status"] == "success"


def test_chat_tracing_production_mode(temp_vector_dir, monkeypatch):
    # Set DEBUG=false using monkeypatch
    monkeypatch.setenv("DEBUG", "false")
    get_settings.cache_clear()

    # Upload & index Doc A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Unique text in Document A about leaves.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    mock_chunk = Chunk(
        chunk_id="docA-c0",
        document_id=doc_a_id,
        source_filename="docA.txt",
        text="Unique text in Document A about leaves.",
        chunk_index=0,
        page=None,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]

        # include_trace=True, but DEBUG=False
        resp = client.post(
            "/chat/query",
            json={"query": "leaves", "top_k": 5, "include_trace": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should NOT contain trace
        assert "trace" not in data

        # Check log file to confirm it was still written
        logs = _get_log_lines(temp_vector_dir)
        assert len(logs) == 1
        assert logs[0]["status"] == "success"


def test_privacy_safe_logging(temp_vector_dir, monkeypatch):
    # 1. First test default LOG_QUERY_PREVIEWS=false behavior
    monkeypatch.setenv("LOG_QUERY_PREVIEWS", "false")
    get_settings.cache_clear()

    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Secret text details.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    mock_chunk = Chunk(
        chunk_id="docA-c0",
        document_id=doc_a_id,
        source_filename="docA.txt",
        text="Secret text details.",
        chunk_index=0,
        page=None,
        block_type="paragraph",
    )

    query_str = "What is the secret context?"
    expected_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]

        resp = client.post(
            "/chat/query",
            json={"query": query_str, "top_k": 5, "include_trace": True},
        )
        assert resp.status_code == 200

    logs = _get_log_lines(temp_vector_dir)
    assert len(logs) == 1
    log_entry = logs[0]

    # Verify query string is NOT in the logs directly
    assert query_str not in log_entry.values()
    assert log_entry["query_hash"] == expected_hash
    assert log_entry["query_length"] == len(query_str)
    assert log_entry["query_preview"] is None

    # Verify chunk text is NOT in retrieved_chunks
    assert "retrieved_chunks" in log_entry
    for chunk_meta in log_entry["retrieved_chunks"]:
        assert "text" not in chunk_meta
        assert chunk_meta["chunk_id"] == "docA-c0"
        assert chunk_meta["score"] == 0.8

    # 2. Test LOG_QUERY_PREVIEWS=true behavior
    monkeypatch.setenv("LOG_QUERY_PREVIEWS", "true")
    get_settings.cache_clear()

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]
        resp = client.post(
            "/chat/query",
            json={"query": "Another secret request?", "top_k": 5, "include_trace": True},
        )

    logs2 = _get_log_lines(temp_vector_dir)
    assert len(logs2) == 2
    log_entry2 = logs2[1]
    assert log_entry2["query_preview"] is not None
    assert "Another" in log_entry2["query_preview"]


def test_tracing_error_logging(temp_vector_dir):
    # Upload & index Doc A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Vivek Shinde leaves information.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    mock_chunk = Chunk(
        chunk_id="docA-c0",
        document_id=doc_a_id,
        source_filename="docA.txt",
        text="Vivek Shinde leaves information.",
        chunk_index=0,
        page=None,
        block_type="paragraph",
    )

    # Mock the LLM generation to throw ValueError (simulates OpenAI timeout/API key failure)
    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]
        with patch("app.services.llm_service.LLMService.generate_answer") as mock_gen:
            mock_gen.side_effect = ValueError("API keys expired or invalid format.")

            resp = client.post(
                "/chat/query",
                json={"query": "error trace query", "top_k": 5, "include_trace": True},
            )
            # Assert custom wrapper catches exception and raises 502
            assert resp.status_code == 502

    # Verify trace indicates status="error" and details written BEFORE propagation
    logs = _get_log_lines(temp_vector_dir)
    assert len(logs) == 1
    log_entry = logs[0]
    assert log_entry["status"] == "error"
    assert log_entry["error_details"] == "API keys expired or invalid format."
    # Secrets should not be present in error_details
    assert "sk-" not in log_entry["error_details"]
