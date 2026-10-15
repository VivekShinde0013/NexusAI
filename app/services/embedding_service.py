from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
import numpy as np
from typing import List
from app.config import get_settings


class EmbeddingService:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
            # Load the heavy ML dependency only when embeddings are needed.
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise ValueError(
                    f"Failed to load sentence-transformers model "
                    f"'{self.model_name}': {str(e)}"
                )
        return self._model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates L2-normalized float32 embeddings for a list of texts."""
        dimension = self.get_dimension()
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)

        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            embeddings = np.array(embeddings, dtype=np.float32)

            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            # L2 normalization for cosine-style similarity (Inner Product)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return embeddings / norms
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {str(e)}")

    def get_dimension(self) -> int:
        if self.model_name == "all-MiniLM-L6-v2":
            return 384
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            return 384


_embedding_service_instance = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service_instance
    if _embedding_service_instance is None:
        settings = get_settings()
        _embedding_service_instance = EmbeddingService(settings.EMBEDDING_MODEL)
    return _embedding_service_instance
