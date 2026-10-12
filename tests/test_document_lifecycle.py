import os
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone
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

    get_settings.cache_clear()

    # Reset vector store to ensure clear index on each test run
    store = get_vector_store()
    store.clear()
    yield
    get_settings.cache_clear()


# --- TDD Verification Cases ---


def test_upload_sets_metadata_status(temp_vector_dir):
    # Upload a document
    txt_path = temp_vector_dir / "raw" / "test.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("Hello World", encoding="utf-8")

    with open(txt_path, "rb") as f:
        resp = client.post(
            "/documents/upload", files={"file": ("test.txt", f, "text/plain")}
        )
    assert resp.status_code == 200
    data = resp.json()
    doc_id = data["document_id"]

    # Verify sidecar JSON contents
    sidecar_path = temp_vector_dir / "raw" / f"{doc_id}.json"
    assert sidecar_path.exists()

    with open(sidecar_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["status"] == "uploaded"
    assert "uploaded_at" in meta
    # Verify timezone-aware format: parsing should succeed and have timezone
    dt = datetime.fromisoformat(meta["uploaded_at"])
    assert dt.tzinfo is not None
    assert meta["stored_filename"] == data["stored_filename"]


def test_index_updates_metadata_status(temp_vector_dir):
    # Upload
    txt_path = temp_vector_dir / "raw" / "test.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("Context to index.", encoding="utf-8")

    with open(txt_path, "rb") as f:
        resp = client.post(
            "/documents/upload", files={"file": ("test.txt", f, "text/plain")}
        )
    doc_id = resp.json()["document_id"]

    # Trigger Indexing
    idx_resp = client.post(f"/documents/{doc_id}/index")
    assert idx_resp.status_code == 200

    # Verify updated metadata
    sidecar_path = temp_vector_dir / "raw" / f"{doc_id}.json"
    with open(sidecar_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["status"] == "indexed"
    assert "indexed_at" in meta
    dt = datetime.fromisoformat(meta["indexed_at"])
    assert dt.tzinfo is not None


def test_list_documents(temp_vector_dir):
    # Upload A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Content A", encoding="utf-8")
    with open(txt_a, "rb") as f:
        client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )

    # Upload B
    txt_b = temp_vector_dir / "raw" / "docB.txt"
    txt_b.write_text("Content B", encoding="utf-8")
    with open(txt_b, "rb") as f:
        resp_b = client.post(
            "/documents/upload", files={"file": ("docB.txt", f, "text/plain")}
        )
    doc_b_id = resp_b.json()["document_id"]

    # Index B only
    client.post(f"/documents/{doc_b_id}/index")

    # List documents
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()

    assert len(docs) == 2
    ids = [d["document_id"] for d in docs]
    assert doc_b_id in ids

    # Check that status values are preserved in list
    for doc in docs:
        if doc["document_id"] == doc_b_id:
            assert doc["status"] == "indexed"
        else:
            assert doc["status"] == "uploaded"


def test_get_document_by_id(temp_vector_dir):
    txt_path = temp_vector_dir / "raw" / "test.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("Content text", encoding="utf-8")
    with open(txt_path, "rb") as f:
        resp = client.post(
            "/documents/upload", files={"file": ("test.txt", f, "text/plain")}
        )
    doc_id = resp.json()["document_id"]

    # Get by valid ID
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["document_id"] == doc_id
    assert get_resp.json()["status"] == "uploaded"

    # Get by invalid UUID format -> 400
    get_invalid = client.get("/documents/invalid-uuid-format")
    assert get_invalid.status_code == 400

    # Get by non-existent UUID -> 404
    get_missing = client.get("/documents/11111111-2222-3333-4444-555555555555")
    assert get_missing.status_code == 404


def test_delete_document(temp_vector_dir):
    # 1. Upload & Index Doc A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Vivek Shinde leaves details.", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]
    stored_filename_a = resp_a.json()["stored_filename"]
    client.post(f"/documents/{doc_a_id}/index")

    # 2. Upload & Index Doc B
    txt_b = temp_vector_dir / "raw" / "docB.txt"
    txt_b.write_text("Other context details.", encoding="utf-8")
    with open(txt_b, "rb") as f:
        resp_b = client.post(
            "/documents/upload", files={"file": ("docB.txt", f, "text/plain")}
        )
    doc_b_id = resp_b.json()["document_id"]
    client.post(f"/documents/{doc_b_id}/index")

    # Verify search returns chunk from A
    search_resp = client.post("/search", json={"query": "leaves", "top_k": 5})
    assert len(search_resp.json()) >= 1
    chunk_ids = [c["chunk_id"] for c in search_resp.json()]
    assert any(doc_a_id in cid for cid in chunk_ids)

    # 3. Delete Doc A
    del_resp = client.delete(f"/documents/{doc_a_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["document_id"] == doc_a_id
    assert del_resp.json()["status"] == "deleted"

    # Assert Filesystem Cleanup
    assert not (temp_vector_dir / "raw" / f"{doc_a_id}.json").exists()
    assert not (temp_vector_dir / "raw" / stored_filename_a).exists()

    # Assert GET /documents contains no Doc A
    list_resp = client.get("/documents")
    assert doc_a_id not in [d["document_id"] for d in list_resp.json()]

    # Assert in-memory Vector store contains no Doc A entries
    store = get_vector_store()
    for chunk in store.chunks:
        assert chunk.document_id != doc_a_id

    # Discard and reload FAISS store from disk to assert persisted index is clean
    store.clear()
    store.load_from_disk_if_empty()
    for chunk in store.chunks:
        assert chunk.document_id != doc_a_id

    # Assert search cannot return chunks from Doc A, but still returns Doc B
    search_resp2 = client.post("/search", json={"query": "leaves", "top_k": 5})
    chunk_ids2 = [c["chunk_id"] for c in search_resp2.json()]
    for cid in chunk_ids2:
        assert doc_a_id not in cid

    # Repeat delete returns 404
    del_repeat = client.delete(f"/documents/{doc_a_id}")
    assert del_repeat.status_code == 404

    # Delete with invalid UUID -> 400
    del_invalid = client.delete("/documents/invalid-uuid-format")
    assert del_invalid.status_code == 400


def test_repeated_indexing_duplicate_prevention(temp_vector_dir):
    # Upload Doc A
    txt_a = temp_vector_dir / "raw" / "docA.txt"
    txt_a.parent.mkdir(parents=True, exist_ok=True)
    txt_a.write_text("Hello repeated index", encoding="utf-8")
    with open(txt_a, "rb") as f:
        resp_a = client.post(
            "/documents/upload", files={"file": ("docA.txt", f, "text/plain")}
        )
    doc_a_id = resp_a.json()["document_id"]

    # Index once
    client.post(f"/documents/{doc_a_id}/index")
    store = get_vector_store()
    initial_count = store.count()
    assert initial_count > 0

    # Index again
    client.post(f"/documents/{doc_a_id}/index")
    # Verify count is replaced (not added/duplicated)
    assert store.count() == initial_count


def test_document_id_consistency_lifecycle(temp_vector_dir):
    """Verifies that upload -> index -> FAISS -> list -> query preserves exact canonical document ID."""
    txt_path = temp_vector_dir / "raw" / "canonical_cv.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("Vivek Shinde knows Python, Java, JavaScript, C++, C, SQL, TypeScript", encoding="utf-8")

    # 1. Upload creates canonical ID
    with open(txt_path, "rb") as f:
        up_resp = client.post("/documents/upload", files={"file": ("canonical_cv.txt", f, "text/plain")})
    assert up_resp.status_code == 200
    canonical_id = up_resp.json()["document_id"]
    assert canonical_id

    # 2. Index preserves exact canonical ID
    idx_resp = client.post(f"/documents/{canonical_id}/index")
    assert idx_resp.status_code == 200
    assert idx_resp.json()["document_id"] == canonical_id

    # 3. FAISS chunks use exact same document ID
    store = get_vector_store()
    matching_chunks = [c for c in store.chunks if c.document_id == canonical_id]
    assert len(matching_chunks) > 0
    for chunk in matching_chunks:
        assert chunk.document_id == canonical_id
        assert chunk.chunk_id.startswith(canonical_id)

    # 4. Documents API returns exact same ID and indexed status
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    matching_docs = [d for d in list_resp.json() if d["document_id"] == canonical_id]
    assert len(matching_docs) == 1
    assert matching_docs[0]["status"] == "indexed"

    # 5. Document-scoped retrieval using returned ID finds indexed chunks
    chat_resp = client.post("/chat/query", json={
        "query": "which programming languages vivek knows",
        "document_ids": [canonical_id],
        "top_k": 5
    })
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert "answer" in chat_data


def test_unindexed_document_cannot_be_queried(temp_vector_dir):
    """Verifies that an unindexed document (status: uploaded) cannot silently behave as a valid chat query target."""
    txt_path = temp_vector_dir / "raw" / "unindexed_doc.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("Unindexed content that is not in FAISS.", encoding="utf-8")

    with open(txt_path, "rb") as f:
        up_resp = client.post("/documents/upload", files={"file": ("unindexed_doc.txt", f, "text/plain")})
    unindexed_id = up_resp.json()["document_id"]

    # Chat query attempting to scope to unindexed document -> returns 400 Bad Request
    chat_resp = client.post("/chat/query", json={
        "query": "test query",
        "document_ids": [unindexed_id],
        "top_k": 5
    })
    assert chat_resp.status_code == 400
    assert "not indexed" in chat_resp.json()["detail"].lower()

