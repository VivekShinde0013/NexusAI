import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from app.services.chunking_service import Chunk
from app.config import get_settings


class VectorStore:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Chunk] = []

    def count(self) -> int:
        return self.index.ntotal

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Adds L2-normalized embeddings for chunks into the index."""
        if not chunks:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of chunks does not match embeddings count.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Incompatible embedding dimension: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        # Skip chunks whose text is empty/whitespace-only
        valid_chunks = []
        valid_indices = []
        for idx, chunk in enumerate(chunks):
            if chunk.text and chunk.text.strip():
                valid_chunks.append(chunk)
                valid_indices.append(idx)

        if not valid_chunks:
            return

        valid_embeddings = embeddings[valid_indices]

        # Add vectors to FAISS index
        self.index.add(valid_embeddings)
        self.chunks.extend(valid_chunks)

    def search(
        self, query_vector: np.ndarray, top_k: int, document_ids: Optional[List[str]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Searches the vector store, optionally filtering by document_ids."""
        self.load_from_disk_if_empty()

        total_vectors = self.count()
        if total_vectors == 0:
            return []

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Incompatible query vector dimension: expected {self.dimension}, got {query_vector.shape[1]}"
            )

        # Handle document filter
        if document_ids is not None:
            if not document_ids:
                # document_ids = [] means return empty results
                return []

            # Exact pre-filtering: isolate matching chunks and reconstruct their vectors
            matching_indices = []
            matching_chunks = []
            for idx, chunk in enumerate(self.chunks):
                if chunk.document_id in document_ids:
                    matching_indices.append(idx)
                    matching_chunks.append(chunk)

            if not matching_chunks:
                return []

            # Reconstruct embeddings
            reconstructed_embeddings = []
            for idx in matching_indices:
                vector = self.index.reconstruct(idx)
                reconstructed_embeddings.append(vector)

            # Build a temporary IndexFlatIP index (MVP Pre-filtering scalability tradeoff)
            temp_index = faiss.IndexFlatIP(self.dimension)
            temp_embeddings = np.array(reconstructed_embeddings, dtype=np.float32)
            temp_index.add(temp_embeddings)

            # Retrieve candidate pool
            k_candidate = min(len(matching_chunks), max(top_k * 3, top_k))
            if k_candidate <= 0:
                return []

            distances, indices = temp_index.search(query_vector, k_candidate)

            candidates = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(matching_chunks):
                    continue
                candidates.append((matching_chunks[idx], float(dist)))

            return self._select_diverse_chunks(candidates, top_k)

        # Unfiltered Search
        # Retrieve candidate pool
        k_candidate = min(total_vectors, max(top_k * 3, top_k))
        if k_candidate <= 0:
            return []

        distances, indices = self.index.search(query_vector, k_candidate)

        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            candidates.append((self.chunks[idx], float(dist)))

        return self._select_diverse_chunks(candidates, top_k)

    @staticmethod
    def _select_diverse_chunks(
        candidates: List[Tuple[Chunk, float]], top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """Selects up to top_k results from candidate pool while balancing document diversity.

        - Pass 1 (Diversity Pass): Pick top-scoring chunk for each unique document.
        - Pass 2 (Score Fill Pass): Fill remaining slots up to top_k from remaining candidates.
        - Pass 3 (Score Order Preservation): Return selected results sorted by similarity score descending.
        """
        if len(candidates) <= top_k:
            return candidates

        selected: List[Tuple[Chunk, float]] = []
        seen_doc_ids = set()

        # Pass 1: Select top chunk for each unique document
        for chunk, score in candidates:
            if chunk.document_id not in seen_doc_ids:
                selected.append((chunk, score))
                seen_doc_ids.add(chunk.document_id)
                if len(selected) == top_k:
                    break

        # Pass 2: Fill remaining slots with highest-scoring unselected candidates
        if len(selected) < top_k:
            selected_ids = {item[0].chunk_id for item in selected}
            for chunk, score in candidates:
                if chunk.chunk_id not in selected_ids:
                    selected.append((chunk, score))
                    selected_ids.add(chunk.chunk_id)
                    if len(selected) == top_k:
                        break

        # Pass 3: Sort by similarity score descending
        selected.sort(key=lambda item: item[1], reverse=True)
        return selected

    def remove_document(self, document_id: str):
        """Removes vectors and metadata entries matching document_id."""
        self.load_from_disk_if_empty()

        total_vectors = self.count()
        if total_vectors == 0:
            return

        remaining_embeddings = []
        remaining_chunks = []

        for idx, chunk in enumerate(self.chunks):
            if chunk.document_id != document_id:
                # Reconstruct vector embedding from FAISS
                vector = self.index.reconstruct(idx)
                remaining_embeddings.append(vector)
                remaining_chunks.append(chunk)

        # Reset
        self.clear()

        # Re-build from remaining
        if remaining_embeddings:
            embeddings_arr = np.array(remaining_embeddings, dtype=np.float32)
            self.index.add(embeddings_arr)
            self.chunks = remaining_chunks

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        faiss.write_index(self.index, str(index_path))

        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=4)

    def load(self, directory: Path):
        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Persisted vector store files not found.")

        index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = [Chunk(**item) for item in chunks_data]

        # Corrupted index size validation
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Vector store corruption: FAISS index size ({index.ntotal}) does not match metadata length ({len(chunks)})."
            )

        self.index = index
        self.chunks = chunks

    def load_from_disk_if_empty(self):
        """Auto-loads persisted index/metadata if memory is empty."""
        if self.count() == 0:
            settings = get_settings()
            store_dir = settings.get_vector_store_path()
            index_path = store_dir / "index.faiss"
            metadata_path = store_dir / "metadata.json"
            if index_path.exists() and metadata_path.exists():
                try:
                    self.load(store_dir)
                except Exception:
                    # Graceful validation check fallback
                    pass


_vector_store_instance = None


def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        from app.services.embedding_service import get_embedding_service

        embedding_service = get_embedding_service()
        dimension = embedding_service.get_dimension()
        _vector_store_instance = VectorStore(dimension=dimension)
    return _vector_store_instance
