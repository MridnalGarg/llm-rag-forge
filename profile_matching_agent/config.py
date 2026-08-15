import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

RESUME_DIR = Path(os.getenv("RESUME_DIR", BASE_DIR / "data" / "resumes"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "data" / "chroma"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", BASE_DIR / "reports"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "40"))
ROUND1_TOP_K = int(os.getenv("ROUND1_TOP_K", "10"))
ROUND2_TOP_K = int(os.getenv("ROUND2_TOP_K", "5"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "3"))

SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".docx", ".txt"}

for directory in (RESUME_DIR, CHROMA_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
