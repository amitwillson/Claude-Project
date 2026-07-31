# Embedding provider abstraction.
#
# Tradeoff note: Anthropic does not currently offer a dedicated embeddings
# endpoint (the Messages API is generation-only), so the default provider
# here is a local, open-source sentence-transformers model
# (all-MiniLM-L6-v2). This costs no API money and keeps embedding fully
# local/offline, at the cost of somewhat lower retrieval quality than a
# large hosted embedding model would give. The EmbeddingProvider interface
# below is intentionally pluggable so a hosted provider (OpenAI, Voyage,
# Cohere, etc.) could be swapped in later without touching the rest of the
# indexing/retrieval pipeline.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""
        ...


class SentenceTransformersEmbeddings(EmbeddingProvider):
    """Local, free, offline embedding provider using sentence-transformers.
    Default and recommended provider for this project."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class AnthropicEmbeddings(EmbeddingProvider):
    """Placeholder provider documenting the tradeoff: Anthropic's API does
    not currently expose an embeddings endpoint. This class exists so the
    interface has a clearly-named "if Anthropic ever ships embeddings, wire
    it up here" seam, and raises immediately if someone tries to use it, so
    the failure is loud rather than a silent no-op."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Anthropic does not currently provide a dedicated embeddings API. "
            "Use SentenceTransformersEmbeddings (default, local, free) or "
            "plug in another hosted embeddings provider via the "
            "EmbeddingProvider interface."
        )

    @property
    def dimension(self) -> int:  # pragma: no cover - unreachable, see __init__
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError


def get_default_provider() -> EmbeddingProvider:
    return SentenceTransformersEmbeddings()
