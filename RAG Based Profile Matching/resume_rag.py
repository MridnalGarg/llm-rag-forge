from pathlib import Path

from document_loader import DocumentLoader
from chunker import ResumeChunker
from metadata import MetadataExtractor
from embedder import Embedder
from vector_store import VectorStore


def main():
    base_dir = Path(__file__).resolve().parent
    resume_dir = base_dir / "resumes"

    loader = DocumentLoader(str(resume_dir))
    chunker = ResumeChunker()
    metadata_extractor = MetadataExtractor()
    embedder = Embedder()
    vector_store = VectorStore()

    documents = loader.load_documents()

    if not documents:
        print("No documents were loaded. Check that the resumes directory contains readable PDF files.")
        return

    all_chunks = []

    for document in documents:

        metadata = metadata_extractor.extract(document)

        chunks = chunker.chunk_document(document)

        # Attach metadata to every chunk
        for chunk in chunks:
            chunk["metadata"] = metadata

        all_chunks.extend(chunks)

    embedded_chunks = embedder.embed_chunks(all_chunks)

    print(f"\nCreated {len(embedded_chunks)} embeddings\n")

    if not embedded_chunks:
        print("No embeddings were generated. Exiting early.")
        return

    #Reset the collection before adding new chunks to avoid duplicates
    vector_store.reset_collection()

    vector_store.add_chunks(
        embedded_chunks
    )

    print("Finished indexing resumes!" + "=" * 10)


if __name__ == "__main__":
    print("Starting the RAG-based profile matching process...\n")
    main()
    