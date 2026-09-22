import os
import tempfile
import json
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings, Settings
from app.services.vector_store import get_vector_store

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

    get_settings.cache_clear()

    # Reset vector store to ensure clear index on each test run
    store = get_vector_store()
    store.clear()
    yield
    get_settings.cache_clear()




# --- Endpoint-level Tests ---


def test_index_document_success(temp_vector_dir):
    # 1. Upload a text document
    txt_path = temp_vector_dir / "raw" / "policy.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(
        "This is an enterprise policy regarding leaves.", encoding="utf-8"
    )

    with open(txt_path, "rb") as f:
        resp = client.post(
            "/documents/upload", files={"file": ("policy.txt", f, "text/plain")}
        )
    assert resp.status_code == 200
    doc_id = resp.json()["document_id"]

    # 2. Trigger index
    index_resp = client.post(f"/documents/{doc_id}/index")
    assert index_resp.status_code == 200
    data = index_resp.json()

    assert data["document_id"] == doc_id
    assert data["chunks_indexed"] == 1
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert data["vector_dimension"] == 384
    assert data["status"] == "indexed"


def test_index_document_missing():
    missing_id = "00000000-0000-0000-0000-000000000000"
    resp = client.post(f"/documents/{missing_id}/index")
    assert resp.status_code == 404
    assert "metadata not found" in resp.json()["detail"].lower()


def test_search_endpoint_success(temp_vector_dir):
    # 1. Upload and index a text document
    txt_path = temp_vector_dir / "raw" / "policy.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("Employees receive 25 annual leave days.", encoding="utf-8")

    with open(txt_path, "rb") as f:
        resp = client.post(
            "/documents/upload", files={"file": ("policy.txt", f, "text/plain")}
        )
    doc_id = resp.json()["document_id"]
    client.post(f"/documents/{doc_id}/index")

    # 2. Query search endpoint
    search_payload = {"query": "annual leave days", "top_k": 2}
    search_resp = client.post("/search", json=search_payload)
    assert search_resp.status_code == 200
    results = search_resp.json()

    assert len(results) == 1  # Capped at ntotal (which is 1)
    hit = results[0]
    assert hit["document_id"] == doc_id
    assert hit["source_filename"] == "policy.txt"
    assert "annual leave" in hit["text"]
    assert "score" in hit
    assert hit["page"] is None
    assert hit["block_type"] == "document"


def test_search_endpoint_parameter_validation():
    # Empty query text
    resp_empty = client.post("/search", json={"query": "   ", "top_k": 5})
    assert resp_empty.status_code == 400
    assert "cannot be empty" in resp_empty.json()["detail"]

    # top_k < 1
    resp_topk = client.post("/search", json={"query": "test query", "top_k": 0})
    assert resp_topk.status_code == 400
    assert "must be at least 1" in resp_topk.json()["detail"]


def test_search_endpoint_empty_store():
    # Calling search when no documents have been indexed should return empty result
    resp = client.post("/search", json={"query": "anything", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json() == []
