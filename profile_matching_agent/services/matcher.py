from collections import defaultdict
import re

from config import TOP_K_RETRIEVAL
from rag.store import ResumeVectorStore


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", value.lower()).strip()


class CandidateMatcher:
    def __init__(self):
        self.store = ResumeVectorStore()

    def search(self, requirements: dict, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        query = self._query_text(requirements)
        result = self.store.search(query, top_k)

        grouped = defaultdict(list)

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            grouped[metadata["candidate_id"]].append({
                "document": doc,
                "metadata": metadata,
                "distance": float(distance),
            })

        candidates = []

        for candidate_id, chunks in grouped.items():
            metadata = chunks[0]["metadata"]
            best_distance = min(c["distance"] for c in chunks)
            semantic = max(0.0, 1.0 - min(best_distance, 1.0))

            candidates.append(
                self._score_candidate(
                    candidate_id,
                    metadata,
                    chunks,
                    requirements,
                    semantic,
                )
            )

        return sorted(
            candidates,
            key=lambda candidate: candidate["score"],
            reverse=True,
        )

    def _query_text(self, requirements: dict) -> str:
        values = []
        for key in (
            "required_skills",
            "preferred_skills",
            "role_requirements",
            "domain_requirements",
        ):
            values.extend(requirements.get(key, []))

        return " ".join(values) or requirements.get(
            "summary",
            "software engineer",
        )

    def _score_candidate(
        self,
        candidate_id,
        metadata,
        chunks,
        requirements,
        semantic,
    ):
        candidate_skills = {
            norm(skill) for skill in metadata.get("skills", [])
        }

        required = {
            norm(skill) for skill in requirements.get("required_skills", [])
        }

        preferred = {
            norm(skill) for skill in requirements.get("preferred_skills", [])
        }

        matched_required = sorted(required & candidate_skills)
        missing_required = sorted(required - candidate_skills)
        matched_preferred = sorted(preferred & candidate_skills)

        must_score = len(matched_required) / max(len(required), 1)
        preferred_score = len(matched_preferred) / max(len(preferred), 1)

        minimum_exp = float(
            requirements.get("minimum_experience_years") or 0
        )
        experience = float(
            metadata.get("experience_years") or 0
        )

        experience_score = (
            1.0
            if minimum_exp == 0
            else min(experience / minimum_exp, 1.0)
        )

        experience_ok = experience >= minimum_exp

        score = (
            0.50 * must_score
            + 0.20 * experience_score
            + 0.20 * semantic
            + 0.10 * preferred_score
        )

        if missing_required:
            score *= 0.82

        if minimum_exp and not experience_ok:
            score *= 0.70

        return {
            "candidate_id": candidate_id,
            "candidate_name": metadata.get(
                "candidate_name",
                candidate_id,
            ),
            "score": round(score * 100, 2),
            "score_breakdown": {
                "must_have_fit": round(must_score * 100, 2),
                "experience_fit": round(experience_score * 100, 2),
                "semantic_fit": round(semantic * 100, 2),
                "preferred_fit": round(preferred_score * 100, 2),
            },
            "metadata": metadata,
            "matched_required": matched_required,
            "missing_required": missing_required,
            "matched_preferred": matched_preferred,
            "experience_ok": experience_ok,
            "matching_evidence": [
                chunk["document"][:700]
                for chunk in chunks[:3]
            ],
        }
