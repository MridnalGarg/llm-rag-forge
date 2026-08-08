import json

try:
    from openai import OpenAI, APITimeoutError
except ImportError:
    OpenAI = None
    APITimeoutError = Exception

from config import OPENROUTER_API_KEY, DEFAULT_MODEL, API_BASE_URL

class LLMClient:

    def __init__(self):
        api_key = OPENROUTER_API_KEY
        self.client = None

        if not api_key or OpenAI is None:
            return

        self.client = OpenAI(
            api_key=api_key,
            base_url=API_BASE_URL,
            timeout=30.0,
            max_retries=2,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "RAG-Based Profile Matching",
            }
        )

    def extract_resume_metadata(self, resume_text):

        prompt = f"""
You are an expert resume parser.

Extract the following information from the resume text.

Return ONLY valid JSON.

{{
    "candidate_name": "",
    "skills": [],
    "education": [],
    "experience_years": 0,
    "summary": ""
}}

If a field is unknown, use an empty string or empty array as appropriate.

Resume:
{resume_text}
"""

        if self.client is None:
            return self._fallback_metadata()

        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_body={
                    "response_format": {"type": "json_object"}
                }
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from model")

            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                print(f"Model returned non-JSON content: {content}")
                raise ValueError(f"Invalid JSON from model: {exc}") from exc
        except (APITimeoutError, TimeoutError, ValueError, Exception) as exc:
            print(f"LLM request failed: {exc}")
            return self._fallback_metadata()

    def _fallback_metadata(self):
        return {
            "candidate_name": "",
            "skills": [],
            "education": [],
            "experience_years": 0,
            "summary": "Unable to extract metadata at the moment."
        }

    def _fallback_job_requirements(self):
        return {
            "job_title": "",
            "required_skills": [],
            "preferred_skills": [],
            "experience_years": 0,
            "summary": "Unable to extract job requirements at the moment."
        }

    def explain_candidate_fit(self, job_requirements: dict, candidate_metadata: dict, matching_chunks: list) -> str:
        """
        Produce a concise recruiter-style explanation of why a candidate is a strong or weak fit.
        """
        if self.client is None:
            return self._fallback_fit_explanation(job_requirements, candidate_metadata)

        prompt = f"""
You are a recruiter reviewing a candidate for a role.

Job Requirements
{json.dumps(job_requirements, indent=2)}

Candidate Metadata
{json.dumps(candidate_metadata, indent=2)}

Top Matching Chunks
{json.dumps(matching_chunks[:3], indent=2)}

Explain in 3 concise sentences why this candidate is a strong or weak fit.
Do not exaggerate.
Mention both strengths and missing requirements.
"""

        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                temperature=0.2,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from model")

            return content.strip()
        except (APITimeoutError, TimeoutError, ValueError, Exception) as exc:
            print(f"LLM explanation failed: {exc}")
            return self._fallback_fit_explanation(job_requirements, candidate_metadata)

    def _fallback_fit_explanation(self, job_requirements: dict, candidate_metadata: dict) -> str:
        required_skills = job_requirements.get("required_skills") or []
        candidate_skills = candidate_metadata.get("skills") or []
        experience_years = candidate_metadata.get("experience_years")

        matches = [skill for skill in required_skills if skill in candidate_skills]
        missing = [skill for skill in required_skills if skill not in candidate_skills]

        if matches and experience_years:
            summary = f"The candidate has {experience_years} years of experience and directly matches skills such as {', '.join(matches)}."
        elif matches:
            summary = f"The candidate directly matches skills such as {', '.join(matches)}."
        else:
            summary = "The candidate does not clearly align with the stated requirements."

        if missing:
            summary += f" Missing requirements include {', '.join(missing)}."

        return summary

    def extract_job_requirements(self, job_description: str) -> dict:
        """
        Parse a Job Description into structured requirements.
        """
        prompt = f"""
            You are an expert job description parser.
            Extract the following information from the job description text.
            Return ONLY valid JSON.

            {{
                "job_title": "",
                "required_skills": [],
                "preferred_skills": [],
                "experience_years": 0,
                "summary": ""
            }}
            Job Description:
            {job_description}
            """

        if self.client is None:
            return self._fallback_job_requirements()

        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_body={
                    "response_format": {"type": "json_object"}
                }
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from model")

            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                print(f"Model returned non-JSON content: {content}")
                raise ValueError(f"Invalid JSON from model: {exc}") from exc
        except (APITimeoutError, TimeoutError, ValueError, Exception) as exc:
            print(f"LLM request failed: {exc}")
            return self._fallback_job_requirements()