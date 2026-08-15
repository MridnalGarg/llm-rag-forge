import uuid
import streamlit as st

from agent.runtime import AgentRuntime
from config import RESUME_DIR
from services.indexer import ResumeIndexer

st.set_page_config(
    page_title="Agentic Profile Matching",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Agentic Profile Matching")
st.caption(
    "LangGraph + hybrid RAG + Gemini free-tier model"
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "runtime" not in st.session_state:
    st.session_state.runtime = AgentRuntime()

if "history" not in st.session_state:
    st.session_state.history = []

if "state" not in st.session_state:
    st.session_state.state = {}

with st.sidebar:
    st.subheader("Resume Index")
    st.write(f"Resume folder: `{RESUME_DIR}`")

    if st.button(
        "Index / Refresh resumes",
        use_container_width=True,
    ):
        with st.spinner("Indexing resumes..."):
            result = ResumeIndexer().index(reset=True)

        st.success(
            f"Indexed {len(result['indexed'])} resumes."
        )

        if result["errors"]:
            st.warning(
                f"{len(result['errors'])} files had errors."
            )

    st.divider()
    st.subheader("Try an example")

    examples = [
        "Find me candidates with React and 3+ years experience",
        "Compare the top 3 matches side by side",
        "Why did John rank higher than Jane?",
        "Add AWS as a must-have and re-rank",
        "Generate interview questions for the top candidate",
        "Run the full 3-round screening",
    ]

    for example in examples:
        if st.button(
            example,
            use_container_width=True,
        ):
            st.session_state.pending_query = example

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input(
    "Ask about candidates, rankings, requirements, or screening..."
)

query = query or st.session_state.pop(
    "pending_query",
    None,
)

if query:
    st.session_state.history.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agent working..."):
            try:
                result = st.session_state.runtime.invoke(
                    query,
                    st.session_state.thread_id,
                    st.session_state.state,
                )

                st.session_state.state = result

                answer = (
                    result.get("answer")
                    or result.get("report")
                    or result.get("comparison")
                    or "Done."
                )

                st.markdown(answer)

                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer,
                })

            except Exception as exc:
                message = f"Agent error: {exc}"
                st.error(message)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": message,
                })

state = st.session_state.state

if state.get("pending_human_feedback"):
    st.divider()
    st.subheader("Human feedback loop")
    st.write(
        "Review the shortlist. Accept it or add a new requirement."
    )

    feedback = st.text_input(
        "Feedback",
        placeholder=(
            "accept / make AWS a must-have / require Kubernetes"
        ),
    )

    if st.button(
        "Submit feedback",
        type="primary",
    ):
        with st.spinner("Applying feedback and re-ranking..."):
            try:
                result = st.session_state.runtime.resume(
                    feedback,
                    st.session_state.thread_id,
                )

                st.session_state.state = result

                answer = (
                    result.get("answer")
                    or result.get("report")
                    or "Updated."
                )

                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer,
                })

                st.rerun()

            except Exception as exc:
                st.error(
                    f"Feedback error: {exc}"
                )

shortlist = state.get("shortlist", [])

if shortlist:
    st.divider()
    st.subheader("Current shortlist")

    for index, candidate in enumerate(
        shortlist,
        start=1,
    ):
        with st.expander(
            f"{index}. "
            f"{candidate.get('candidate_name')} — "
            f"{candidate.get('score', 0):.1f}%"
        ):
            st.write(
                "**Matched must-haves:**",
                ", ".join(
                    candidate.get(
                        "matched_required",
                        [],
                    )
                ) or "None",
            )

            st.write(
                "**Missing must-haves:**",
                ", ".join(
                    candidate.get(
                        "missing_required",
                        [],
                    )
                ) or "None",
            )

            st.write(
                "**Score breakdown:**",
                candidate.get(
                    "score_breakdown",
                    {},
                ),
            )

            analysis = candidate.get(
                "analysis",
                {},
            )

            if analysis:
                st.write(
                    "**Strengths:**",
                    analysis.get(
                        "strengths",
                        [],
                    ),
                )
                st.write(
                    "**Gaps:**",
                    analysis.get(
                        "gaps",
                        [],
                    ),
                )
                st.write(
                    "**Evidence:**",
                    analysis.get(
                        "evidence",
                        [],
                    ),
                )
                st.write(
                    "**Improvement suggestions:**",
                    analysis.get(
                        "improvement_suggestions",
                        [],
                    ),
                )
