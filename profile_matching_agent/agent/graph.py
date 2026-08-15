from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    apply_feedback_node,
    analyze_shortlist_node,
    compare_node,
    extract_requirements_node,
    general_node,
    generate_report_node,
    human_feedback_node,
    interview_node,
    parse_input,
    parse_jd,
    rank_candidates_node,
    screening_node,
    search_resumes_node,
    why_ranked_node,
)
from agent.state import AgentState


def after_parse(state: AgentState) -> str:
    return state.get("intent", "search")


def after_search(state: AgentState) -> str:
    if state.get("intent") == "screening":
        return "screening"
    return "rank"


def after_screening(state: AgentState) -> str:
    if state.get("final_recommendation"):
        return "end"
    return "screening"


def after_feedback(state: AgentState) -> str:
    if state.get("feedback", "").lower().strip() in {
        "accept",
        "approve",
        "done",
        "continue",
    }:
        return "end"

    return "rerank"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_input", parse_input)
    graph.add_node("parse_jd", parse_jd)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("analyze_shortlist", analyze_shortlist_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("human_feedback", human_feedback_node)
    graph.add_node("apply_feedback", apply_feedback_node)
    graph.add_node("compare", compare_node)
    graph.add_node("why_ranked", why_ranked_node)
    graph.add_node("interview", interview_node)
    graph.add_node("screening", screening_node)
    graph.add_node("general", general_node)

    graph.add_edge(START, "parse_input")

    graph.add_conditional_edges(
        "parse_input",
        after_parse,
        {
            "search": "parse_jd",
            "refine": "extract_requirements",
            "compare": "compare",
            "why_ranked": "why_ranked",
            "interview": "interview",
            "screening": "parse_jd",
            "general": "general",
        },
    )

    graph.add_edge(
        "parse_jd",
        "extract_requirements",
    )

    graph.add_edge(
        "extract_requirements",
        "search_resumes",
    )

    graph.add_conditional_edges(
        "search_resumes",
        after_search,
        {
            "screening": "screening",
            "rank": "rank_candidates",
        },
    )

    graph.add_edge(
        "rank_candidates",
        "analyze_shortlist",
    )

    graph.add_edge(
        "analyze_shortlist",
        "generate_report",
    )

    graph.add_edge(
        "generate_report",
        "human_feedback",
    )

    graph.add_edge(
        "human_feedback",
        "apply_feedback",
    )

    graph.add_conditional_edges(
        "apply_feedback",
        after_feedback,
        {
            "rerank": "search_resumes",
            "end": END,
        },
    )

    graph.add_edge("compare", END)
    graph.add_edge("why_ranked", END)
    graph.add_edge("interview", END)
    graph.add_edge("general", END)

    graph.add_conditional_edges(
        "screening",
        after_screening,
        {
            "screening": "screening",
            "end": END,
        },
    )

    return graph.compile(checkpointer=MemorySaver())
