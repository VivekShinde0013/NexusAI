import tempfile
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


# --- TDD Test Cases ---


def test_search_with_document_filter(temp_vector_dir):
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

    # Upload & index Doc B
    txt_b = temp_vector_dir / "raw" / "docB.txt"
    txt_b.write_text("Unique text in Document B about leaves.", encoding="utf-8")
    with open(txt_b, "rb") as f:
        resp_b = client.post(
            "/documents/upload", files={"file": ("docB.txt", f, "text/plain")}
        )
    doc_b_id = resp_b.json()["document_id"]
    client.post(f"/documents/{doc_b_id}/index")

    # Search with Doc A filter
    search_payload = {
        "query": "leaves",
        "top_k": 5,
        "document_ids": [doc_a_id],
    }
    resp = client.post("/search", json=search_payload)
    assert resp.status_code == 200
    results = resp.json()

    # Verify only Doc A chunk is returned
    for r in results:
        assert r["document_id"] == doc_a_id


def test_chat_with_document_filter(temp_vector_dir):
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

    # Upload & index Doc B
    txt_b = temp_vector_dir / "raw" / "docB.txt"
    txt_b.write_text("Other context info about leaves.", encoding="utf-8")
    with open(txt_b, "rb") as f:
        resp_b = client.post(
            "/documents/upload", files={"file": ("docB.txt", f, "text/plain")}
        )
    doc_b_id = resp_b.json()["document_id"]
    client.post(f"/documents/{doc_b_id}/index")

    # RAG query with Doc A filter only
    mock_chunk = Chunk(
        chunk_id="docA-c0",
        document_id=doc_a_id,
        source_filename="docA.txt",
        text="Vivek Shinde leaves information.",
        chunk_index=0,
        page=None,
        block_type="paragraph",
    )

    with patch("app.services.vector_store.VectorStore.search") as mock_search:
        mock_search.return_value = [(mock_chunk, 0.8)]

        chat_payload = {
            "query": "What do we know about Vivek?",
            "top_k": 5,
            "document_ids": [doc_a_id],
        }
        resp = client.post("/chat/query", json=chat_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["confidence"] in ["high", "medium"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_filename"] == "docA.txt"



def test_search_with_multiple_filters(temp_vector_dir):
    # Upload A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Leaves content A.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    # Upload B
    txt_b = temp_vector_dir / "raw" / "docB.txt"
    txt_b.write_text("Leaves content B.", encoding="utf-8")
    with open(txt_b, "rb") as f:
        resp_b = client.post(
            "/documents/upload", files={"file": ("docB.txt", f, "text/plain")}
        )
    doc_b_id = resp_b.json()["document_id"]
    client.post(f"/documents/{doc_b_id}/index")

    # Upload C (Should NOT be returned)
    txt_c = temp_vector_dir / "raw" / "docC.txt"
    txt_c.write_text("Leaves content C.", encoding="utf-8")
    with open(txt_c, "rb") as f:
        resp_c = client.post(
            "/documents/upload", files={"file": ("docC.txt", f, "text/plain")}
        )
    doc_c_id = resp_c.json()["document_id"]
    client.post(f"/documents/{doc_c_id}/index")

    # Search with Doc A and Doc B filter
    resp = client.post(
        "/search", json={"query": "leaves", "top_k": 10, "document_ids": [doc_a_id, doc_b_id]}
    )
    assert resp.status_code == 200
    results = resp.json()

    returned_ids = {r["document_id"] for r in results}
    assert doc_c_id not in returned_ids
    assert doc_a_id in returned_ids or doc_b_id in returned_ids


def test_filtering_invalid_uuid():
    # Pass invalid UUID format in filters
    resp = client.post(
        "/search", json={"query": "test", "top_k": 5, "document_ids": ["invalid-uuid"]}
    )
    assert resp.status_code == 400
    assert "format" in resp.json()["detail"].lower()


def test_filtering_nonexistent_uuid():
    # Pass valid UUID format that doesn't exist in metadata
    nonexistent_id = "00000000-1111-2222-3333-444455556666"
    resp = client.post(
        "/search", json={"query": "test", "top_k": 5, "document_ids": [nonexistent_id]}
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_filtering_empty_document_ids(temp_vector_dir):
    # Upload A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Leaves content A.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    # document_ids=[] means return nothing
    resp_search = client.post(
        "/search", json={"query": "leaves", "top_k": 5, "document_ids": []}
    )
    assert resp_search.status_code == 200
    assert resp_search.json() == []

    resp_chat = client.post(
        "/chat/query", json={"query": "leaves", "top_k": 5, "document_ids": []}
    )
    assert resp_chat.status_code == 200
    assert resp_chat.json()["answer"] == "I could not find information about this in the provided documents."
    assert resp_chat.json()["sources"] == []


def test_filtering_duplicate_document_ids(temp_vector_dir):
    # Upload A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Leaves content A.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    # Pass duplicates: [A, A]
    resp = client.post(
        "/search", json={"query": "leaves", "top_k": 5, "document_ids": [doc_a_id, doc_a_id]}
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1  # Deduplicated and returns exactly 1 chunk


def test_filtering_deleted_document(temp_vector_dir):
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

    # Delete Doc A
    del_resp = client.delete(f"/documents/{doc_a_id}")
    assert del_resp.status_code == 200

    # Assert search filter by deleted Doc A returns 404 (metadata not found)
    resp = client.post(
        "/search", json={"query": "leaves", "top_k": 5, "document_ids": [doc_a_id]}
    )
    assert resp.status_code == 404

    # Assert persisted FAISS vector store reloaded contains no vectors or metadata for Doc A
    store = get_vector_store()
    store.clear()
    store.load_from_disk_if_empty()
    assert store.count() == 0
    assert len(store.chunks) == 0


def test_filtering_unfiltered_fallback(temp_vector_dir):
    # Upload & Index A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Leaves content A.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    client.post(f"/documents/{doc_a_id}/index")

    # document_ids=null or omitted means search the entire store
    resp = client.post(
        "/search", json={"query": "leaves", "top_k": 5}  # document_ids is null/omitted
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["document_id"] == doc_a_id
