import os
import uuid
from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException

# Supported extensions (lowercase with dots)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}

# Map of extension to expected MIME types
ALLOWED_MIME_TYPES = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    },
    ".txt": {"text/plain"},
    ".csv": {
        "text/csv",
        "application/csv",
        "text/comma-separated-values",
        "application/vnd.ms-excel",
    },
}


def validate_file_extension(filename: str) -> str:
    """Extracts and validates the file extension.

    Returns the lowercase extension (including the dot).
    """
    if not filename:
        raise HTTPException(
            status_code=400, detail="Filename is missing or empty"
        )

    _, ext = os.path.splitext(filename)
    ext = ext.lower().strip()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions are: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


def validate_mime_type(content_type: Optional[str], extension: str) -> None:
    """Validates the MIME type if it is provided and recognizable.

    Does not reject generic fallback MIME types like application/octet-stream.
    """
    if not content_type:
        return

    content_type_lower = content_type.lower().strip()
    # Allow common fallback/generic/binary MIME types
    if content_type_lower in {
        "application/octet-stream",
        "binary/octet-stream",
        "",
    }:
        return

    expected_mimes = ALLOWED_MIME_TYPES.get(extension, set())
    if expected_mimes and content_type_lower not in expected_mimes:
        # Collect all supported MIME types across all extensions
        all_supported_mimes = set()
        for mimes in ALLOWED_MIME_TYPES.values():
            all_supported_mimes.update(mimes)

        # If it's not a generic fallback and not supported anywhere, raise error
        if content_type_lower not in all_supported_mimes:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported MIME type '{content_type}'.",
            )


def generate_safe_filename(extension: str) -> str:
    """Generates a secure unique stored filename using UUID."""
    return f"{uuid.uuid4()}{extension}"


def save_file(
    file_content: bytes, destination_dir: Path, filename: str
) -> Tuple[str, int]:
    """Writes file content to disk inside destination_dir.

    Returns the absolute path as string and total size in bytes.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_path = destination_dir / filename

    with open(file_path, "wb") as f:
        f.write(file_content)

    return str(file_path.resolve()), len(file_content)
