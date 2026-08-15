import hashlib

import chromadb

from config import CHROMA_DIR
from rag.embeddings import LocalEmbedder


COLLECTION_NAME = "resume_chunks"


class ResumeVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )
        self.embedder = LocalEmbedder()

    def _get_collection(self):
        """Always return the current Chroma collection."""
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        """Safely recreate the resume collection."""
        try:
            self.client.delete_collection(
                name=COLLECTION_NAME
            )
        except Exception:
            pass

        # Force creation of the new collection.
        self._get_collection()

    def add_chunks(
        self,
        candidate_id: str,
        chunks: list[str],
        metadata: dict,
    ):
        if not chunks:
            return

        collection = self._get_collection()

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            digest = hashlib.sha1(
                f"{candidate_id}:{index}:{chunk}".encode(
                    "utf-8"
                )
            ).hexdigest()

            ids.append(digest)
            documents.append(chunk)

            metadatas.append({
                **metadata,
                "candidate_id": candidate_id,
                "chunk_index": index,
            })

        embeddings = self.embedder.embed(documents)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(
        self,
        query: str,
        top_k: int = 40,
    ) -> dict:
        collection = self._get_collection()

        # Handle an empty collection gracefully.
        if collection.count() == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        embedding = self.embedder.embed_one(query)

        # Don't request more results than exist.
        n_results = min(
            top_k,
            collection.count(),
        )

        return collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )