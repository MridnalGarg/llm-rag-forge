import json
from typing import Any

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to .env."
            )

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL

    def text(self, prompt: str, temperature: float = 0.2) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="text/plain",
            ),
        )
        return (response.text or "").strip()

    def json(self, prompt: str) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return json.loads((response.text or "").strip())

    def extract_requirements(self, jd: str) -> dict:
        prompt = f'''
You are an expert technical recruiter and job-description parser.
Return ONLY valid JSON.

Schema:
{{
  "job_title": "",
  "summary": "",
  "required_skills": [],
  "preferred_skills": [],
  "minimum_experience_years": 0,
  "maximum_experience_years": null,
  "education": [],
  "role_requirements": [],
  "domain_requirements": [],
  "constraints": []
}}

Rules:
- Put explicit must-have requirements in required_skills.
- Put explicitly preferred/nice-to-have requirements in preferred_skills.
- Infer numeric experience only when the text supports it.
- Do not invent requirements.

Job description:
{jd}
'''
        return self.json(prompt)

    def classify_intent(self, query: str, has_existing_results: bool) -> dict:
        prompt = f'''
Classify this recruiting assistant request.
Return ONLY JSON.

Schema:
{{
  "intent": "search|refine|compare|why_ranked|interview|screening|general"
}}

Existing candidate results: {has_existing_results}
User request:
{query}
'''
        return self.json(prompt)

    def generate_candidate_analysis(self, requirements: dict, candidate: dict) -> dict:
        prompt = f'''
Analyze this candidate only from the supplied evidence.
Return ONLY JSON.

Schema:
{{
  "strengths": [],
  "gaps": [],
  "matched_requirements": [],
  "missing_requirements": [],
  "evidence": [],
  "borderline": false,
  "improvement_suggestions": [],
  "summary": ""
}}

Requirements:
{json.dumps(requirements, indent=2)}

Candidate:
{json.dumps(candidate, indent=2)}
'''
        return self.json(prompt)

    def compare(self, candidates: list[dict]) -> str:
        return self.text(
            "Compare these candidates for a recruiter. Use only supplied facts. "
            "Explain who is strongest, where each wins, and important gaps.\n\n"
            + json.dumps(candidates, indent=2)
        )

    def interview_questions(self, candidate: dict, requirements: dict) -> list[str]:
        prompt = f'''
Create 8 screening interview questions.
Return ONLY JSON:
{{"questions": []}}

Include technical depth, evidence validation, missing requirements and role fit.

Candidate:
{json.dumps(candidate, indent=2)}

Requirements:
{json.dumps(requirements, indent=2)}
'''
        return self.json(prompt).get("questions", [])

    def report(self, requirements: dict, candidates: list[dict], round_name: str) -> str:
        prompt = f'''
Generate a recruiter-facing match report for {round_name}.
Use only supplied data.
Do not make a final employment decision on behalf of a human.

Include:
- ranking
- score
- strengths
- gaps
- evidence
- ranking rationale
- borderline improvement suggestions
- recommended next screening step

Requirements:
{json.dumps(requirements, indent=2)}

Candidates:
{json.dumps(candidates, indent=2)}
'''
        return self.text(prompt, temperature=0.15)
