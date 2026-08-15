import json
import re

from langgraph.types import interrupt

from agent.state import AgentState
from config import FINAL_TOP_K, ROUND1_TOP_K, ROUND2_TOP_K, TOP_K_RETRIEVAL
from services.llm_client import GeminiService
from services.matcher import CandidateMatcher

llm = GeminiService()
matcher = CandidateMatcher()


def parse_input(state: AgentState) -> dict:
    result = llm.classify_intent(
        state.get("user_query", ""),
        bool(state.get("candidate_pool")),
    )
    return {"intent": result.get("intent", "search")}


def parse_jd(state: AgentState) -> dict:
    return {"job_description": state.get("user_query", "")}


def extract_requirements_node(state: AgentState) -> dict:
    if state.get("intent") == "refine":
        previous = state.get("requirements", {})
        refinement = llm.extract_requirements(
            state.get("user_query", "")
        )
        return {
            "previous_requirements": previous,
            "requirements": merge_requirements(
                previous,
                refinement,
            ),
        }

    requirements = llm.extract_requirements(
        state.get("job_description")
        or state.get("user_query", "")
    )

    return {
        "previous_requirements": state.get(
            "requirements",
            {},
        ),
        "requirements": requirements,
    }


def search_resumes_node(state: AgentState) -> dict:
    candidates = matcher.search(
        state.get("requirements", {}),
        TOP_K_RETRIEVAL,
    )
    return {
        "candidate_pool": candidates,
        "rankings": candidates,
    }


def rank_candidates_node(state: AgentState) -> dict:
    rankings = sorted(
        state.get("candidate_pool", []),
        key=lambda item: item.get("score", 0),
        reverse=True,
    )

    return {
        "rankings": rankings,
        "shortlist": rankings[:ROUND1_TOP_K],
    }


def analyze_shortlist_node(state: AgentState) -> dict:
    enriched = []

    for candidate in state.get("shortlist", []):
        try:
            analysis = llm.generate_candidate_analysis(
                state.get("requirements", {}),
                candidate,
            )
        except Exception as exc:
            analysis = {
                "summary": f"Analysis unavailable: {exc}",
                "strengths": [],
                "gaps": [],
                "evidence": [],
                "improvement_suggestions": [],
            }

        enriched.append({
            **candidate,
            "analysis": analysis,
        })

    return {
        "shortlist": enriched,
        "rankings": enriched,
    }


def generate_report_node(state: AgentState) -> dict:
    report = llm.report(
        state.get("requirements", {}),
        state.get("shortlist", []),
        "Round 1",
    )

    return {
        "report": report,
        "answer": report,
        "pending_human_feedback": True,
    }


def human_feedback_node(state: AgentState) -> dict:
    feedback = interrupt({
        "type": "human_feedback",
        "message": (
            "Review the shortlist. Reply 'accept' to finish, "
            "or provide new criteria to re-rank."
        ),
        "shortlist": state.get("shortlist", []),
    })

    return {
        "feedback": str(feedback),
        "pending_human_feedback": False,
    }


def apply_feedback_node(state: AgentState) -> dict:
    feedback = state.get("feedback", "").strip()

    if feedback.lower() in {
        "accept",
        "approve",
        "done",
        "continue",
    }:
        return {
            "answer": "Shortlist accepted by recruiter.",
            "pending_human_feedback": False,
        }

    previous = state.get("requirements", {})
    refinement = llm.extract_requirements(feedback)
    merged = merge_requirements(
        previous,
        refinement,
    )

    old_rank = {
        candidate["candidate_id"]: index + 1
        for index, candidate
        in enumerate(state.get("rankings", []))
    }

    new_candidates = matcher.search(
        merged,
        TOP_K_RETRIEVAL,
    )

    new_rank = {
        candidate["candidate_id"]: index + 1
        for index, candidate
        in enumerate(new_candidates)
    }

    explanation = ranking_change_explanation(
        old_rank,
        new_rank,
        new_candidates,
    )

    return {
        "previous_requirements": previous,
        "requirements": merged,
        "candidate_pool": new_candidates,
        "rankings": new_candidates,
        "shortlist": new_candidates[:ROUND1_TOP_K],
        "ranking_change_explanation": explanation,
        "answer": explanation,
        "pending_human_feedback": True,
    }


def compare_node(state: AgentState) -> dict:
    candidates = state.get("candidate_pool", [])
    selected = candidates[:3]

    comparison = llm.compare(selected)

    return {
        "comparison": comparison,
        "answer": comparison,
    }


def why_ranked_node(state: AgentState) -> dict:
    query = state.get("user_query", "").lower()
    rankings = state.get("rankings", [])

    mentioned = [
        candidate
        for candidate in rankings
        if candidate.get("candidate_name", "").lower()
        in query
    ]

    selected = (
        mentioned[:2]
        if len(mentioned) >= 2
        else rankings[:2]
    )

    if len(selected) < 2:
        return {
            "answer": (
                "I need at least two candidates in the current "
                "ranking to explain the difference."
            )
        }

    higher, lower = sorted(
        selected,
        key=lambda item: item.get("score", 0),
        reverse=True,
    )[:2]

    answer = (
        f"{higher['candidate_name']} ranked higher "
        f"({higher['score']} vs {lower['score']}).\n\n"
        f"Higher candidate score breakdown: "
        f"{json.dumps(higher.get('score_breakdown', {}))}\n"
        f"Matched must-haves: "
        f"{higher.get('matched_required', [])}\n"
        f"Missing must-haves: "
        f"{higher.get('missing_required', [])}\n\n"
        f"Lower candidate score breakdown: "
        f"{json.dumps(lower.get('score_breakdown', {}))}\n"
        f"Matched must-haves: "
        f"{lower.get('matched_required', [])}\n"
        f"Missing must-haves: "
        f"{lower.get('missing_required', [])}"
    )

    return {"answer": answer}


def interview_node(state: AgentState) -> dict:
    query = state.get("user_query", "").lower()
    candidates = state.get("candidate_pool", [])

    candidate = next(
        (
            item
            for item in candidates
            if item.get("candidate_name", "").lower()
            in query
        ),
        candidates[0] if candidates else None,
    )

    if not candidate:
        return {
            "answer": (
                "No candidate is available in the current shortlist."
            )
        }

    questions = llm.interview_questions(
        candidate,
        state.get("requirements", {}),
    )

    return {
        "interview_questions": questions,
        "answer": "\n".join(
            f"{index + 1}. {question}"
            for index, question in enumerate(questions)
        ),
    }


def screening_node(state: AgentState) -> dict:
    round_number = int(
        state.get("screening_round", 1)
    )

    candidates = state.get("candidate_pool", [])

    if round_number == 1:
        selected = candidates[:ROUND1_TOP_K]
    elif round_number == 2:
        selected = candidates[:ROUND2_TOP_K]
    else:
        selected = candidates[:FINAL_TOP_K]

    enriched = []

    for candidate in selected:
        analysis = llm.generate_candidate_analysis(
            state.get("requirements", {}),
            candidate,
        )
        enriched.append({
            **candidate,
            "analysis": analysis,
        })

    history = state.get(
        "round_history",
        [],
    ) + [{
        "round": round_number,
        "candidates": enriched,
    }]

    result = {
        "shortlist": enriched,
        "rankings": enriched,
        "round_history": history,
    }

    if round_number < 3:
        result["screening_round"] = round_number + 1
    else:
        report = llm.report(
            state.get("requirements", {}),
            enriched,
            "Final screening round",
        )
        result.update({
            "final_recommendation": report,
            "report": report,
            "answer": report,
        })

    return result


def general_node(state: AgentState) -> dict:
    prompt = (
        "You are a recruiting assistant. Answer briefly.\n"
        f"Current requirements: {json.dumps(state.get('requirements', {}))}\n"
        f"User: {state.get('user_query', '')}"
    )
    return {"answer": llm.text(prompt)}


def merge_requirements(old: dict, new: dict) -> dict:
    merged = dict(old or {})

    list_keys = [
        "required_skills",
        "preferred_skills",
        "education",
        "role_requirements",
        "domain_requirements",
        "constraints",
    ]

    for key in list_keys:
        values = []

        for item in (
            (old or {}).get(key, [])
            + (new or {}).get(key, [])
        ):
            if item and item not in values:
                values.append(item)

        merged[key] = values

    for key in [
        "job_title",
        "summary",
        "minimum_experience_years",
        "maximum_experience_years",
    ]:
        value = (new or {}).get(key)
        if value not in (None, "", [], 0):
            merged[key] = value

    return merged


def ranking_change_explanation(
    old_rank: dict,
    new_rank: dict,
    new_candidates: list[dict],
) -> str:
    lines = [
        "Ranking updated based on the new criteria:"
    ]

    for candidate in new_candidates[:10]:
        cid = candidate["candidate_id"]
        before = old_rank.get(cid, "new")
        after = new_rank.get(cid, "new")

        if before != after:
            lines.append(
                f"- {candidate['candidate_name']}: "
                f"{before} -> {after} "
                f"(score {candidate['score']})"
            )

    if len(lines) == 1:
        lines.append(
            "The ordering did not materially change."
        )

    return "\n".join(lines)
