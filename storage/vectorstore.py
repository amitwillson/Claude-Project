# Local persistent Chroma vector store wrapper. Designed for incremental
# adds — add_chunks() upserts by chunk_id, so re-running extraction on new
# documents only adds their chunks rather than rebuilding the whole index.
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

COLLECTION_NAME = "ir_circulars"


class VectorStore:
    def __init__(self, persist_dir: str | Path = "data/chroma_db"):
        import chromadb  # lazy import

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def add_chunks(
        self,
        chunk_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not chunk_ids:
            return
        # Chroma's upsert is idempotent by id -> safe for incremental re-runs.
        self._collection.upsert(
            ids=list(chunk_ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def query(self, embedding: Sequence[float], top_n: int = 5) -> list[dict]:
        result = self._collection.query(query_embeddings=[list(embedding)], n_results=top_n)
        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i in range(len(ids)):
            hits.append(
                {
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()
