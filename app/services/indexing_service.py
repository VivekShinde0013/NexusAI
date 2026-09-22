from typing import List
from app.services.chunking_service import Chunk
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store
from app.config import get_settings


def index_chunks(document_id: str, chunks: List[Chunk]) -> int:
    """Embeds, indexes, and persists chunk vectors.

    Removes existing document vectors before indexing to prevent duplication.
    """
    store = get_vector_store()
    embedding_service = get_embedding_service()
    settings = get_settings()

    # 1. Clean existing vectors for document_id to prevent duplicates
    store.remove_document(document_id)

    # 2. Filter empty text chunks
    valid_chunks = [
        c for c in chunks if c.text and c.text.strip()
    ]
    if not valid_chunks:
        # Save index with previously existing document_id deleted
        store.save(settings.get_vector_store_path())
        return 0

    # 3. Generate L2-normalized embeddings
    texts = [c.text for c in valid_chunks]
    embeddings = embedding_service.generate_embeddings(texts)

    # 4. Add to vector store and persist
    store.add_chunks(valid_chunks, embeddings)
    store.save(settings.get_vector_store_path())

    return len(valid_chunks)
