import chromadb
from chromadb.config import Settings

from config import CHROMA_DB_PATH


class VectorStore:
    """
    Handles storing and retrieving embeddings using ChromaDB.
    """

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name="resume_chunks"
        )

    def _normalize_metadata_value(self, value):
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value)
        return value

    def add_chunks(self, embedded_chunks):

        for chunk in embedded_chunks:

            metadata = {}

            for key, value in chunk["metadata"].items():
                metadata[key] = self._normalize_metadata_value(value)

            self.collection.add(

                ids=[chunk["id"]],

                documents=[chunk["document"]],

                embeddings=[
                    chunk["embedding"].tolist()
                ],

                metadatas=[metadata]
            )

    def search(self, query_embedding, top_k=10):

        results = self.collection.query(

            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results=top_k
        )

        return results

    '''
    Reset the collection by deleting it and creating a new one.
    This is useful for clearing the database and starting fresh.
    '''
    def reset_collection(self):
        self.client.delete_collection("resume_chunks")
        self.collection = self.client.get_or_create_collection(
            name="resume_chunks"
        )