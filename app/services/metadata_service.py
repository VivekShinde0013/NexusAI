import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import get_settings


def _get_metadata_path(document_id: str) -> Path:
    settings = get_settings()
    upload_path = settings.get_upload_path()
    return upload_path / f"{document_id}.json"


def create_metadata(
    document_id: str,
    original_filename: str,
    stored_filename: str,
    file_type: str,
    file_size: int,
    mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates and saves the initial sidecar metadata JSON file for an upload."""
    settings = get_settings()
    metadata = {
        "document_id": document_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "file_type": file_type,
        "file_size": file_size,
        "mime_type": mime_type,
        "status": "uploaded",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "indexed_at": None,
    }

    meta_path = _get_metadata_path(document_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata


def update_status(document_id: str, status: str) -> Dict[str, Any]:
    """Updates the status and index timestamp of a document's sidecar metadata."""
    meta_path = _get_metadata_path(document_id)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found for document {document_id}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["status"] = status
    if status == "indexed":
        metadata["indexed_at"] = datetime.now(timezone.utc).isoformat()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata


def get_metadata(document_id: str) -> Optional[Dict[str, Any]]:
    """Loads the sidecar metadata dictionary for a specific document."""
    meta_path = _get_metadata_path(document_id)
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_all_metadata() -> List[Dict[str, Any]]:
    """Lists all document metadata records currently saved in the uploads directory."""
    settings = get_settings()
    upload_path = settings.get_upload_path()
    if not upload_path.exists():
        return []

    records = []
    for file in upload_path.iterdir():
        if file.suffix == ".json":
            # Exclude other metadata.json files (e.g. vector store mappings if they reside there)
            # Each document metadata file is named precisely <uuid>.json
            import uuid
            try:
                uuid.UUID(file.stem)
            except ValueError:
                continue

            try:
                with open(file, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except Exception:
                continue

    return records


def delete_document_files(document_id: str) -> bool:
    """Deletes the physical raw file and the sidecar JSON metadata from disk."""
    metadata = get_metadata(document_id)
    if not metadata:
        return False

    settings = get_settings()
    upload_path = settings.get_upload_path()
    stored_filename = metadata.get("stored_filename")

    # 1. Delete physical raw file
    if stored_filename:
        raw_file_path = upload_path / stored_filename
        if raw_file_path.exists():
            os.remove(raw_file_path)

    # 2. Delete sidecar JSON
    meta_path = _get_metadata_path(document_id)
    if meta_path.exists():
        os.remove(meta_path)

    return True
