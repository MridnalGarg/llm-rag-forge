from typing import Annotated, Any, Literal, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    user_query: str
    intent: Literal[
        "search",
        "refine",
        "compare",
        "why_ranked",
        "interview",
        "screening",
        "general",
    ]

    job_description: str
    requirements: dict[str, Any]
    previous_requirements: dict[str, Any]

    candidate_pool: list[dict[str, Any]]
    shortlist: list[dict[str, Any]]
    rankings: list[dict[str, Any]]

    comparison: str
    interview_questions: list[str]
    report: str

    screening_round: int
    round_history: list[dict[str, Any]]
    final_recommendation: str

    feedback: str
    pending_human_feedback: bool

    answer: str
    ranking_change_explanation: str
