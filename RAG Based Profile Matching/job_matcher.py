from collections import defaultdict

from embedder import Embedder
from llm_client import LLMClient
from vector_store import VectorStore
from skills import KNOWN_SKILLS


class JobMatcher:
    """
    Matches resumes against a job description using:
    - Semantic Search
    - Hybrid Keyword Matching
    - Metadata Filtering
    """

    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.llm = self._build_llm_client()

    def _build_llm_client(self):
        try:
            return LLMClient()
        except Exception as exc:
            print(f"LLM client unavailable: {exc}")
            return None

    def match(self, job_description: str, top_k: int = 10):

        # Step 1
        query_embedding = self.embedder.embed_text(job_description)

        # Step 2
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Step 3
        candidates = self.group_by_candidate(search_results)

        # Step 4
        if self.llm is not None:
            try:
                job_requirements = self.llm.extract_job_requirements(job_description)
            except Exception as exc:
                print(f"Job requirement extraction failed: {exc}")
                job_requirements = {}
        else:
            job_requirements = {}

        required_skills = job_requirements.get("required_skills") or self.extract_required_skills(job_description)

        # Step 5
        ranked_candidates = self.rank_candidates(
            candidates,
            required_skills,
            job_requirements=job_requirements
        )

        return {
            "job_description": job_description,
            "top_matches": ranked_candidates
        }

    '''
    Create grouped dictionary of candidates with their associated documents, metadata, and distances.
    '''

    def group_by_candidate(self, results):

        grouped = defaultdict(list)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            candidate = metadata["candidate_name"]

            grouped[candidate].append(
                {
                    "document": doc,
                    "metadata": metadata,
                    "distance": distance
                }
            )

        return grouped

    '''
    Extract required skills from the job description using a simple keyword search.
    '''
    def extract_required_skills(self, job_description):

        text = job_description.lower()

        required = []

        for skill in KNOWN_SKILLS:

            if skill.lower() in text:
                required.append(skill)

        return required

    '''
    Rank candidates based on a combination of semantic similarity and keyword matching.
    '''

    def rank_candidates(self, candidates, required_skills, job_requirements=None):

        if not candidates:
            return []

        ranked = []

        for candidate_name, chunks in candidates.items():

            metadata = chunks[0]["metadata"]

            candidate_skills = metadata.get(
                "skills",
                ""
            ).split(",")

            candidate_skills = [
                skill.strip()
                for skill in candidate_skills
            ]

            matched_skills = []

            for skill in required_skills:

                if skill in candidate_skills:
                    matched_skills.append(skill)

            semantic_score = (
                1 - min(
                    chunk["distance"]
                    for chunk in chunks
                )
            )

            keyword_score = (
                len(matched_skills)
                / max(len(required_skills), 1)
            )

            final_score = (
                semantic_score * 70
                + keyword_score * 30
            )

            ranked.append(
                {
                    "candidate_name": candidate_name,
                    "metadata": metadata,
                    "resume_path": metadata.get("source") or metadata.get("filename") or "unknown",
                    "match_score": round(final_score),
                    "matched_skills": matched_skills,
                    "relevant_excerpts": [
                        chunk["document"][:300]
                        for chunk in chunks[:2]
                    ],
                    "reasoning": self.generate_reasoning(
                        semantic_score,
                        matched_skills,
                        metadata,
                        job_requirements=job_requirements,
                        matching_chunks=chunks
                    )
                }
            )

        ranked.sort(
            key=lambda x: x["match_score"],
            reverse=True
        )

        return ranked

    '''
    Generate reasoning for the match based on semantic score, matched skills, and metadata.
    '''
    def generate_reasoning(
        self,
        semantic_score,
        matched_skills,
        metadata,
        job_requirements=None,
        matching_chunks=None
    ):

        if self.llm is None:
            return self._fallback_reasoning(semantic_score, matched_skills, metadata)

        try:
            explanation = self.llm.explain_candidate_fit(
                job_requirements or {},
                metadata,
                matching_chunks or []
            )
            if explanation:
                return explanation
        except Exception as exc:
            print(f"Reasoning generation failed: {exc}")

        return self._fallback_reasoning(semantic_score, matched_skills, metadata)

    def _fallback_reasoning(self, semantic_score, matched_skills, metadata):
        reasons = []

        if semantic_score > 0.80:
            reasons.append("Excellent semantic match.")
        elif semantic_score > 0.65:
            reasons.append("Good semantic similarity.")

        if matched_skills:
            reasons.append("Matched skills: " + ", ".join(matched_skills))

        if metadata.get("experience_years"):
            reasons.append(f"{metadata['experience_years']} years of experience.")

        return " ".join(reasons)

    def filter_candidates(
        self,
        candidates,
        minimum_experience
    ):

        filtered = {}

        for candidate_name, chunks in candidates.items():

            metadata = chunks[0]["metadata"]

            experience_years = metadata.get(
                "experience_years",
                0
            )

            if experience_years >= minimum_experience:
                filtered[candidate_name] = chunks

        return filtered