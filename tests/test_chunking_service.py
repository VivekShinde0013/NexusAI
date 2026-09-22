import os
import tempfile
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings, Settings
from app.services.document_extractor import ExtractedDocument, ExtractedBlock
from app.services.chunking_service import chunk_document, Chunk

client = TestClient(app)


@pytest.fixture
def temp_upload_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def override_settings(temp_upload_dir, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(temp_upload_dir))
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "10000000")

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- Unit Tests ---


def test_chunk_document_short_text():
    doc = ExtractedDocument(
        document_id="doc-123",
        source_filename="test.txt",
        file_type="txt",
        blocks=[
            ExtractedBlock(
                type="document", index=1, text="Short text content."
            )
        ],
    )
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc-123-c0"
    assert chunks[0].text == "Short text content."
    assert chunks[0].chunk_index == 0
    assert chunks[0].page is None
    assert chunks[0].block_type == "document"


def test_chunk_document_long_text_and_overlap():
    text = "word1 " * 50  # 300 characters
    doc = ExtractedDocument(
        document_id="doc-123",
        source_filename="test.txt",
        file_type="txt",
        blocks=[ExtractedBlock(type="document", index=1, text=text)],
    )
    # With chunk_size=100 and chunk_overlap=20, it should produce multiple chunks
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"doc-123-c{idx}"
        assert c.chunk_index == idx
        # Verify approximate size limit
        assert len(c.text) <= 100


def test_chunk_document_word_boundaries():
    # Length of words: "hello" is 5, "world" is 5.
    text = "hello world split boundary word"
    doc = ExtractedDocument(
        document_id="doc-123",
        source_filename="test.txt",
        file_type="txt",
        blocks=[ExtractedBlock(type="document", index=1, text=text)],
    )
    # With size=13 (enough for "hello world"), it should split at space, not inside "split"
    chunks = chunk_document(doc, chunk_size=13, chunk_overlap=0)
    assert len(chunks) >= 2
    assert chunks[0].text == "hello world"
    assert "split" in chunks[1].text


def test_chunk_document_unbroken_string_fallback():
    # Long unbroken string of 'a's (no whitespaces)
    text = "a" * 100
    doc = ExtractedDocument(
        document_id="doc-123",
        source_filename="test.txt",
        file_type="txt",
        blocks=[ExtractedBlock(type="document", index=1, text=text)],
    )
    # Split size 30, overlap 10. Should split strictly on character boundaries and make forward progress
    chunks = chunk_document(doc, chunk_size=30, chunk_overlap=10)
    assert len(chunks) == 5  # 0-30, 20-50, 40-70, 60-90, 80-100
    assert chunks[0].text == "a" * 30
    assert chunks[1].text == "a" * 30
    assert chunks[4].text == "a" * 20


def test_chunk_document_metadata_and_ids():
    pdf_doc = ExtractedDocument(
        document_id="doc-pdf",
        source_filename="report.pdf",
        file_type="pdf",
        blocks=[
            ExtractedBlock(type="page", index=5, text="PDF Page 5 text."),
            ExtractedBlock(type="page", index=6, text="PDF Page 6 text."),
        ],
    )
    chunks = chunk_document(pdf_doc, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 2
    # Verify deterministic global sequential indexes
    assert chunks[0].chunk_id == "doc-pdf-c0"
    assert chunks[0].page == 5
    assert chunks[0].block_type == "page"
    assert chunks[0].metadata["block_index"] == 5

    assert chunks[1].chunk_id == "doc-pdf-c1"
    assert chunks[1].page == 6
    assert chunks[1].block_type == "page"
    assert chunks[1].metadata["block_index"] == 6


def test_chunk_document_distinguishable_blocks():
    # Verify we do not combine unrelated blocks across boundaries
    doc = ExtractedDocument(
        document_id="doc-docx",
        source_filename="test.docx",
        file_type="docx",
        blocks=[
            ExtractedBlock(type="paragraph", index=1, text="Paragraph 1 text."),
            ExtractedBlock(type="paragraph", index=2, text="Paragraph 2 text."),
        ],
    )
    # Even if chunk_size is 500, we should get 2 separate chunks to preserve block boundaries
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0].text == "Paragraph 1 text."
    assert chunks[1].text == "Paragraph 2 text."
    assert chunks[0].block_type == "paragraph"
    assert chunks[1].block_type == "paragraph"


def test_chunk_document_invalid_config_rejected():
    doc = ExtractedDocument(
        document_id="doc-123",
        source_filename="test.txt",
        file_type="txt",
        blocks=[ExtractedBlock(type="document", index=1, text="valid text")],
    )
    # Overlap >= size
    with pytest.raises(ValueError, match="overlap must be strictly less"):
        chunk_document(doc, chunk_size=50, chunk_overlap=50)

    # Size negative
    with pytest.raises(ValueError, match="chunk_size must be between"):
        chunk_document(doc, chunk_size=0, chunk_overlap=5)

    # Overlap negative
    with pytest.raises(ValueError, match="chunk_overlap must be between"):
        chunk_document(doc, chunk_size=100, chunk_overlap=-1)


# --- API Integration Tests ---


def test_api_chunking_with_overrides(temp_upload_dir):
    # 1. Setup a valid text document and upload it
    txt_path = temp_upload_dir / "test.txt"
    txt_path.write_text("Hello. This is split chunking test.", encoding="utf-8")

    with open(txt_path, "rb") as f:
        resp = client.post(
            "/documents/upload", files={"file": ("test.txt", f, "text/plain")}
        )
    assert resp.status_code == 200
    doc_id = resp.json()["document_id"]

    # 2. Call chunking endpoint with valid overrides
    chunk_resp = client.post(
        f"/documents/{doc_id}/chunks?chunk_size=10&chunk_overlap=2"
    )
    assert chunk_resp.status_code == 200
    chunks = chunk_resp.json()
    assert len(chunks) > 1
    assert chunks[0]["chunk_id"] == f"{doc_id}-c0"
    assert chunks[0]["document_id"] == doc_id
    assert chunks[0]["source_filename"] == "test.txt"

    # 3. Call chunking endpoint with invalid overrides (overlap >= size)
    err_resp = client.post(
        f"/documents/{doc_id}/chunks?chunk_size=10&chunk_overlap=10"
    )
    assert err_resp.status_code == 400
    assert "strictly less than" in err_resp.json()["detail"]

    # 4. Call chunking endpoint with invalid overrides (size too large)
    err_resp_large = client.post(
        f"/documents/{doc_id}/chunks?chunk_size=20000&chunk_overlap=50"
    )
    assert err_resp_large.status_code == 400
    assert "between 1 and 10,000" in err_resp_large.json()["detail"]


def test_api_chunking_missing_document():
    missing_id = "00000000-0000-0000-0000-000000000000"
    resp = client.post(f"/documents/{missing_id}/chunks")
    assert resp.status_code == 404
    assert "metadata not found" in resp.json()["detail"].lower()


def test_api_chunking_invalid_uuid():
    resp = client.post("/documents/invalid-uuid-format/chunks")
    assert resp.status_code == 400
    assert "Invalid document ID format" in resp.json()["detail"]
