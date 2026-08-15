from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import build_graph


class AgentRuntime:
    def __init__(self):
        self.graph = build_graph()

    def invoke(
        self,
        query: str,
        thread_id: str,
        current_state: dict | None = None,
    ):
        state = dict(current_state or {})
        state["user_query"] = query
        state["messages"] = state.get(
            "messages",
            [],
        ) + [HumanMessage(content=query)]

        return self.graph.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            },
        )

    def resume(
        self,
        feedback: str,
        thread_id: str,
    ):
        return self.graph.invoke(
            Command(resume=feedback),
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            },
        )
