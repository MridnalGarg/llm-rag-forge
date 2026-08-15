from config import RESUME_DIR
from filesystem.file_tools import list_files, read_file
from rag.chunker import chunk_text
from rag.metadata import extract_metadata
from rag.store import ResumeVectorStore


class ResumeIndexer:
    def __init__(self):
        self.store = ResumeVectorStore()

    def index(self, reset: bool = False) -> dict:
        if reset:
            self.store.reset()

        files = list_files(str(RESUME_DIR))
        indexed = []
        errors = []

        for item in files:
            result = read_file(item["filepath"])
            if result.get("status") == "error":
                errors.append(result)
                continue

            metadata = extract_metadata(result["content"], item["filename"])
            candidate_id = metadata["candidate_name"].lower().replace(" ", "_")
            chunks = chunk_text(result["content"])

            self.store.add_chunks(candidate_id, chunks, metadata)

            indexed.append({
                "candidate_id": candidate_id,
                "candidate_name": metadata["candidate_name"],
                "experience_years": metadata["experience_years"],
                "skills": metadata["skills"],
                "source_file": item["filename"],
                "chunks": len(chunks),
            })

        return {"indexed": indexed, "errors": errors}
