# llm-rag-forge

## Quick start

1. Create and activate a Python virtual environment.
2. Install the dependencies from the project requirements file:
   - pip install -r "RAG Based Profile Matching/requirements.txt"
3. Place PDF resumes in the resumes folder inside the project directory.
4. Run the indexing workflow:
   - python "RAG Based Profile Matching/resume_rag.py"
5. Run the matcher:
   - python -c "import sys; sys.path.insert(0, r'RAG Based Profile Matching'); from job_matcher import JobMatcher; jm = JobMatcher(); print(jm.match('Python and AWS role', top_k=5))"

6. Run Streamlit App
   - streamlit run main.py

The project degrades gracefully when optional AI packages are not installed, so it can still be imported and exercised in a minimal environment.