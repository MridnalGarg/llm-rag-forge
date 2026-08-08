import re
from datetime import datetime
from llm_client import LLMClient

class MetadataExtractor:
    """
    Rule-based metadata extraction from resumes.

    Extracts:
    - Candidate name
    - Email
    - Phone
    - LinkedIn
    - GitHub
    - Skills
    - Education
    - Approximate years of experience
    """

    KNOWN_SKILLS = {
        # Programming
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "go",
        "rust",
        "php",
        "ruby",

        # Backend
        "django",
        "flask",
        "fastapi",
        "spring",
        "spring boot",
        "node.js",
        "express",

        # Frontend
        "react",
        "angular",
        "vue",
        "next.js",
        "html",
        "css",

        # Databases
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "redis",
        "elasticsearch",

        # Cloud / DevOps
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "github actions",

        # Data / ML
        "machine learning",
        "deep learning",
        "nlp",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "spark",

        # GenAI
        "llm",
        "rag",
        "langchain",
        "llamaindex",
        "chromadb",
        "pinecone",
        "hugging face",
    }

    DEGREE_KEYWORDS = (
        "b.tech",
        "btech",
        "b.e",
        "bachelor",
        "m.tech",
        "mtech",
        "m.e",
        "master",
        "mba",
        "phd",
        "ph.d",
        "b.sc",
        "m.sc",
    )

    def llm_extract(self, text: str) -> dict:
        llmclient = LLMClient()
        llm_metadata = llmclient.extract_resume_metadata(text)

        return llm_metadata

    def rule_extract(self, document: dict) -> dict:
        text = document.get("text", "")

        return {
            "candidate_name": self.extract_name(text),
            "email": self.extract_email(text),
            "phone": self.extract_phone(text),
            "linkedin": self.extract_linkedin(text),
            "github": self.extract_github(text),
            "skills": self.extract_skills(text),
            "education": self.extract_education(text),
            "experience_years": self.extract_experience_years(text),
        }

    def extract(self, document: dict) -> dict:
        rule_metadata = self.rule_extract(document)
        llm_metadata = self.llm_extract(document.get("text", ""))

        final_metadata = self.merge_metadata(rule_metadata, llm_metadata)

        return final_metadata

    def extract_name(self, text: str):
        """
        Simple heuristic:
        the candidate's name is usually one of the first non-empty lines.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines[:5]:

            # Skip obvious contact information
            if "@" in line:
                continue

            if "linkedin" in line.lower():
                continue

            if "github" in line.lower():
                continue

            # Avoid very long lines
            if len(line.split()) > 6:
                continue

            # Name should contain mostly alphabetic characters
            cleaned = re.sub(r"[^A-Za-z\s.'-]", "", line).strip()

            if cleaned and len(cleaned.split()) >= 2:
                return cleaned

        return None

    def extract_email(self, text: str):
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

        match = re.search(pattern, text)

        return match.group(0) if match else None

    def extract_phone(self, text: str):
        """
        Supports common phone formats, including international numbers.
        """

        pattern = (
            r"(?<!\d)"
            r"(?:\+\d{1,3}[\s.-]?)?"
            r"(?:\(?\d{2,4}\)?[\s.-]?)?"
            r"\d{3,5}[\s.-]?\d{4,5}"
            r"(?!\d)"
        )

        matches = re.findall(pattern, text)

        for match in matches:
            digits = re.sub(r"\D", "", match)

            # Most real phone numbers contain roughly 10-15 digits
            if 10 <= len(digits) <= 15:
                return match.strip()

        return None

    def extract_linkedin(self, text: str):
        pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?"

        match = re.search(pattern, text, re.IGNORECASE)

        return match.group(0) if match else None

    def extract_github(self, text: str):
        pattern = r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+/?"

        match = re.search(pattern, text, re.IGNORECASE)

        return match.group(0) if match else None

    def extract_skills(self, text: str):
        """
        Search for known skills using boundary-aware matching.
        """

        found_skills = []

        for skill in self.KNOWN_SKILLS:

            pattern = (
                r"(?<![A-Za-z0-9])"
                + re.escape(skill)
                + r"(?![A-Za-z0-9])"
            )

            if re.search(pattern, text, re.IGNORECASE):
                found_skills.append(skill)

        return sorted(
            skill.title()
            if skill not in {"aws", "gcp", "sql", "nlp", "llm", "rag"}
            else skill.upper()
            for skill in found_skills
        )

    def extract_education(self, text: str):
        """
        Phase-1 heuristic:
        find lines containing common degree names.
        """

        education = []

        for line in text.splitlines():
            clean_line = line.strip()

            if not clean_line:
                continue

            lower_line = clean_line.lower()

            if any(
                degree in lower_line
                for degree in self.DEGREE_KEYWORDS
            ):
                education.append(clean_line)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(education))

    def extract_experience_years(self, text: str):
        """
        Phase-1 heuristic.

        First look for explicit phrases such as:
            "5 years of experience"
            "7+ years experience"

        If none exist, attempt to estimate from year ranges.
        """

        explicit_pattern = (
            r"\b(\d{1,2})(?:\+)?\s+years?"
            r"(?:\s+of)?\s+experience\b"
        )

        matches = re.findall(
            explicit_pattern,
            text,
            re.IGNORECASE,
        )

        if matches:
            return max(int(year) for year in matches)

        return self._estimate_experience_from_years(text)

    def _estimate_experience_from_years(self, text: str):
        """
        Very simple fallback.

        Example:
            2019 - 2023
            2023 - Present

        This is an approximation only.
        """

        current_year = datetime.now().year

        pattern = (
            r"\b(19\d{2}|20\d{2})\b"
            r"\s*(?:-|–|—|to)\s*"
            r"\b(19\d{2}|20\d{2}|present|current)\b"
        )

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE,
        )

        durations = []

        for start, end in matches:
            start_year = int(start)

            if end.lower() in {"present", "current"}:
                end_year = current_year
            else:
                end_year = int(end)

            duration = end_year - start_year

            if 0 <= duration <= 50:
                durations.append(duration)

        if not durations:
            return None

        return sum(durations)

    def merge_metadata(self, rule_data, llm_data):
        merged_data = rule_data.copy()

        for key, value in llm_data.items():
            if key not in merged_data or not merged_data[key]:
                merged_data[key] = value

        return merged_data