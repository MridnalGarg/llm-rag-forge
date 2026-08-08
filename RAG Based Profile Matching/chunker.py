from pathlib import Path

"""
Intelligent resume chunker.

1. Split resumes by semantic sections.
2. Split large sections into overlapping chunks.
3. Attach metadata for retrieval.
"""
class ResumeChunker:

    SECTION_HEADERS = {
        "summary",
        "objective",
        "education",
        "experience",
        "work experience",
        "employment",
        "projects",
        "skills",
        "technical skills",
        "certifications",
        "achievements",
        "internships",
        "publications",
        "languages",
    }

    def __init__(
        self,
        max_words=600,
        overlap_words=50,
    ):
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk_document(self, document, metadata=None):
        """
        Input:
        {
            path,
            filename,
            text
        }

        Output:
        List[dict]
        """

        sections = self._split_into_sections(document)

        chunks = []

        for section_name, section_text in sections:

            section_chunks = self._split_large_section(section_text)

            for idx, chunk_text in enumerate(section_chunks, start=1):

                chunk_id = (
                    f"{Path(document['filename']).stem}"
                    f"_{section_name.lower().replace(' ', '_')}"
                    f"_{idx}"
                )

                chunk_record = {
                    "chunk_id": chunk_id,
                    "filename": document["filename"],
                    "source": document["path"],
                    "section": section_name,
                    "content": chunk_text,
                }

                if metadata:
                    chunk_record["metadata"] = metadata

                chunks.append(chunk_record)

        return chunks

    def _split_into_sections(self, document):

        lines = document["text"].splitlines()

        sections = []

        current_section = "General"

        current_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            lower = line.lower()

            if lower in self.SECTION_HEADERS:

                if current_lines:
                    sections.append(
                        (
                            current_section,
                            "\n".join(current_lines),
                        )
                    )

                current_section = line
                current_lines = []

            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                (
                    current_section,
                    "\n".join(current_lines),
                )
            )

        return sections

    def _split_large_section(self, text):

        words = text.split()

        if len(words) <= self.max_words:
            return [text]

        chunks = []

        step = self.max_words - self.overlap_words

        for start in range(0, len(words), step):

            end = start + self.max_words

            chunk = " ".join(words[start:end])

            chunks.append(chunk)

            if end >= len(words):
                break

        return chunks