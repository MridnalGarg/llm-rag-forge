from langchain_core.tools import tool

from services.llm_client import GeminiService
from services.matcher import CandidateMatcher

_llm = None
_matcher = None


def llm_service():
    global _llm
    if _llm is None:
        _llm = GeminiService()
    return _llm


def matcher():
    global _matcher
    if _matcher is None:
        _matcher = CandidateMatcher()
    return _matcher


@tool
def extract_requirements(jd: str) -> dict:
    """Parse a JD into must-have and nice-to-have requirements."""
    return llm_service().extract_requirements(jd)


@tool
def search_resumes(requirements: dict, top_k: int = 40) -> list[dict]:
    """Search resumes using semantic retrieval and metadata-aware ranking."""
    return matcher().search(requirements, top_k)


@tool
def compare_candidates(
    candidate_ids: list[str],
    candidates: list[dict],
) -> str:
    """Compare candidates head-to-head."""
    selected = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_id") in set(candidate_ids)
    ]
    return llm_service().compare(selected)


@tool
def generate_interview_questions(
    candidate_id: str,
    candidates: list[dict],
    requirements: dict,
) -> list[str]:
    """Generate screening questions for one candidate."""
    candidate = next(
        (
            item
            for item in candidates
            if item.get("candidate_id") == candidate_id
        ),
        None,
    )

    if not candidate:
        return [f"Candidate '{candidate_id}' was not found."]

    return llm_service().interview_questions(
        candidate,
        requirements,
    )


@tool
def generate_match_report(
    requirements: dict,
    candidates: list[dict],
    round_name: str,
) -> str:
    """Generate an explainable recruiter-facing report."""
    return llm_service().report(
        requirements,
        candidates,
        round_name,
    )
