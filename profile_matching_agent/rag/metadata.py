import re

COMMON_SKILLS = [
    "python", "java", "javascript", "typescript", "react", "angular", "vue",
    "node.js", "nodejs", "spring boot", "spring", "microservices", "rest",
    "graphql", "sql", "postgresql", "mysql", "mongodb", "redis", "kafka",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform",
    "git", "linux", "pandas", "numpy", "pytorch", "tensorflow", "fastapi",
    "django", "flask", "c++", "c#", ".net", "spark", "hadoop", "airflow",
]


def extract_metadata(text: str, filename: str) -> dict:
    lower = text.lower()
    skills = [skill for skill in COMMON_SKILLS if skill in lower]

    experience = 0.0
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience",
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+in\s+",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, lower)
        if matches:
            experience = max(float(item) for item in matches)
            break

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_name = ""

    for line in lines[:12]:
        if len(line.split()) in (2, 3) and not any(ch.isdigit() for ch in line):
            if "resume" not in line.lower() and "curriculum" not in line.lower():
                candidate_name = line
                break

    if not candidate_name:
        candidate_name = PathLike(filename)

    return {
        "candidate_name": candidate_name,
        "skills": skills,
        "experience_years": experience,
        "source_file": filename,
        "summary": " ".join(lines[:5])[:800],
    }


def PathLike(filename: str) -> str:
    return filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
