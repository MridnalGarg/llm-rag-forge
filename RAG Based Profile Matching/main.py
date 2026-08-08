import streamlit as st

from job_matcher import JobMatcher


def run_streamlit_app() -> None:
    st.set_page_config(
        page_title="Resume Matcher",
        page_icon="📄",
        layout="wide",
    )

    st.title("Resume Matcher")
    st.write(
        "Provide a job description and select the number of top resumes to retrieve from the indexed resume database."
    )

    with st.form(key="resume_match_form"):
        job_description = st.text_area(
            "Job Description (JD)",
            help="Paste the job description or role requirements here.",
            height=250,
        )

        top_n = st.number_input(
            "Top N resumes",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Select how many candidates should be returned.",
        )

        submit_button = st.form_submit_button("Find Matches")

    if submit_button:
        if not job_description.strip():
            st.warning("Please enter a job description before submitting.")
            return

        with st.spinner("Matching resumes..."):
            matcher = JobMatcher()
            result = matcher.match(job_description, top_k=top_n)

        matches = result.get("top_matches", [])

        if not matches:
            st.info("No matching resumes were found. Try a different description or verify the index.")
            return

        st.success(f"Found {len(matches)} matching resume(s)")

        for rank, match in enumerate(matches, start=1):
            with st.expander(f"{rank}. {match['candidate_name']} — Score: {match['match_score']}"):
                st.markdown(f"**Resume:** {match.get('resume_path', 'unknown')}")
                st.markdown(f"**Matched skills:** {', '.join(match.get('matched_skills', [])) or 'None'}")
                st.markdown(f"**Reasoning:** {match.get('reasoning', 'No reasoning available.')}")

                excerpts = match.get("relevant_excerpts", [])
                if excerpts:
                    st.markdown("**Relevant excerpts from resume:**")
                    for excerpt in excerpts:
                        st.write(excerpt)

                metadata = match.get("metadata")
                if metadata:
                    st.write(metadata)


if __name__ == "__main__":
    run_streamlit_app()
