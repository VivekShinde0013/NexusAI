import os
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings, Settings

client = TestClient(app)


@pytest.fixture
def temp_upload_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def override_settings(temp_upload_dir, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(temp_upload_dir))
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1024")  # 1 KB

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_upload_pdf(temp_upload_dir):
    content = b"%PDF-1.4 dummy pdf content"
    response = client.post(
        "/documents/upload",
        files={"file": ("company_policy.pdf", content, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["original_filename"] == "company_policy.pdf"
    assert data["stored_filename"] != "company_policy.pdf"
    assert data["file_type"] == "pdf"
    assert data["file_size"] == len(content)
    assert data["status"] == "uploaded"

    # Verify saved file
    saved_path = temp_upload_dir / data["stored_filename"]
    assert saved_path.exists()
    assert saved_path.read_bytes() == content


def test_upload_txt(temp_upload_dir):
    content = b"simple txt document text content"
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["file_type"] == "txt"
    assert data["file_size"] == len(content)

    saved_path = temp_upload_dir / data["stored_filename"]
    assert saved_path.exists()


def test_upload_csv(temp_upload_dir):
    content = b"col1,col2\nval1,val2"
    response = client.post(
        "/documents/upload",
        files={"file": ("data.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["file_type"] == "csv"

    saved_path = temp_upload_dir / data["stored_filename"]
    assert saved_path.exists()


def test_upload_docx(temp_upload_dir):
    content = b"fake docx content structure"
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "document.docx",
                content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["file_type"] == "docx"

    saved_path = temp_upload_dir / data["stored_filename"]
    assert saved_path.exists()


def test_upload_generic_mime(temp_upload_dir):
    # Allow common fallback MIME types like application/octet-stream
    content = b"%PDF-1.4 dummy pdf"
    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", content, "application/octet-stream")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["file_type"] == "pdf"


def test_upload_unsupported_extension():
    content = b"dummy png content"
    response = client.post(
        "/documents/upload",
        files={"file": ("image.png", content, "image/png")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_empty_file():
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_upload_oversized_file():
    # MAX_UPLOAD_SIZE is overridden to 1024 bytes (1 KB)
    content = b"a" * 1025
    response = client.post(
        "/documents/upload",
        files={"file": ("large.txt", content, "text/plain")},
    )
    assert response.status_code == 413
    assert "exceeds the limit" in response.json()["detail"]


def test_upload_mismatched_unsupported_mime():
    content = b"dummy content"
    # Sending a totally unrecognized/unsupported MIME type (e.g. image/jpeg)
    response = client.post(
        "/documents/upload",
        files={"file": ("doc.pdf", content, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unsupported MIME type" in response.json()["detail"]
