import re

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from config import EMBEDDING_MODEL


class Embedder:
    """
    Responsible for converting text into vector embeddings.
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        if SentenceTransformer is None:
            self.model = None
            print("sentence-transformers is not installed; using a lightweight fallback embedding strategy.")
            return

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def embed_text(self, text: str):
        """
        Generate embedding for a single piece of text.

        Returns:
            numpy.ndarray
        """

        if self.model is None:
            tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
            if not tokens:
                return []

            vocab = sorted(set(tokens))
            vector = [0.0] * len(vocab)
            for token in tokens:
                try:
                    vector[vocab.index(token)] += 1.0
                except ValueError:
                    continue
            scale = max(1.0, len(tokens))
            return [value / scale for value in vector]

        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def embed_chunks(self, chunks):
        """
        Convert all chunks into embeddings.

        Returns:
            List[dict]
        """

        embedded_chunks = []

        for chunk in chunks:

            vector = self.embed_text(
                chunk["content"]
            )

            embedded_chunks.append(
                {
                    "id": chunk["chunk_id"],
                    "embedding": vector,
                    "document": chunk["content"],
                    "metadata": {
                        **chunk["metadata"],
                        "section": chunk["section"],
                        "filename": chunk["filename"],
                        "source": chunk["source"]
                    }
                }
            )

        return embedded_chunks