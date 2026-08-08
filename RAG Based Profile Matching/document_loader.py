from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

"""
Loads PDF resumes and extracts text.
"""
class DocumentLoader:
    

    def __init__(self, resume_directory: str):
        self.resume_directory = Path(resume_directory)

    def load_documents(self):
        """
        Returns: List[dict]
        """

        documents = []

        if PdfReader is None:
            print("pypdf is not installed; no PDF resumes can be loaded.")
            return documents

        pdf_files = self.resume_directory.glob("*.pdf")

        for pdf in pdf_files:

            try:
                reader = PdfReader(pdf)

                text = ""

                for page in reader.pages:
                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"

                documents.append(
                    {
                        "path": str(pdf),
                        "filename": pdf.name,
                        "text": text.strip(),
                    }
                )

            except Exception as e:

                print(f"Failed to load {pdf.name}: {e}")

        return documents