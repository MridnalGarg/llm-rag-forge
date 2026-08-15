from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import pypdf
import docx

from config import BASE_DIR, SUPPORTED_RESUME_EXTENSIONS


def resolve_path(filepath: str) -> Path:
    path = Path(filepath).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (BASE_DIR / path).resolve()


def read_file(filepath: str) -> dict:
    path = resolve_path(filepath)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {filepath}"}

    try:
        ext = path.suffix.lower()
        if ext == ".txt":
            content = path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            reader = pypdf.PdfReader(str(path))
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            document = docx.Document(str(path))
            content = "\n".join(p.text for p in document.paragraphs)
        else:
            return {"status": "error", "message": f"Unsupported extension: {ext}"}

        stat = path.stat()
        return {
            "status": "success",
            "filepath": str(path),
            "filename": path.name,
            "content": content.strip(),
            "metadata": {
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            },
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def list_files(directory: str, extension: Optional[str] = None) -> list[dict]:
    directory_path = resolve_path(directory)
    if not directory_path.exists():
        return [{"status": "error", "message": f"Directory not found: {directory}"}]

    result = []
    for path in directory_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_RESUME_EXTENSIONS:
            continue
        if extension and path.suffix.lower() != extension.lower():
            continue
        stat = path.stat()
        result.append({
            "filename": path.name,
            "filepath": str(path),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return sorted(result, key=lambda item: item["filename"].lower())


def search_in_file(filepath: str, keyword: str) -> dict:
    result = read_file(filepath)
    if result.get("status") == "error":
        return result

    lines = result["content"].splitlines()
    needle = keyword.lower()
    matches = []
    for idx, line in enumerate(lines):
        if needle in line.lower():
            matches.append({
                "line_number": idx + 1,
                "matched_line": line.strip(),
                "context": "\n".join(lines[max(0, idx - 1): min(len(lines), idx + 2)]).strip(),
            })

    return {
        "status": "success",
        "filepath": result["filepath"],
        "keyword": keyword,
        "total_matches": len(matches),
        "matches": matches,
    }


def write_file(filepath: str, content: str) -> dict:
    path = resolve_path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "status": "success",
        "filepath": str(path),
        "bytes_written": len(content.encode("utf-8")),
    }
