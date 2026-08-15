# Agentic Profile Matching

LangGraph-based conversational resume matching agent combining the architecture of the user's two previous projects:

- LLM-Powered-File-System-Assistant
- llm-rag-forge / RAG Based Profile Matching

## Free AI requirement

Generative AI uses **Google Gemini API free-tier models only**. The default is `gemini-2.5-flash-lite`.

Resume embeddings use the free local `all-MiniLM-L6-v2` model, so indexing does not require a paid embedding API.

No OpenAI, Anthropic, OpenRouter, or other paid model is required.

## Assignment coverage

### Part A
- LangGraph agent state
- Conversation history
- Job requirement state
- Candidate shortlist and reasoning
- File-system tools
- RAG search
- extract_requirements
- compare_candidates
- generate_interview_questions
- Human feedback loop

### Part B
- Natural-language candidate search
- Candidate comparison
- Ranking explanation
- Requirement refinement
- Automatic re-ranking

### Part C
- Round 1: 100 -> top 10
- Round 2: top 10 -> top 5
- Final round: top 3
- Evidence-backed strengths and gaps
- Borderline candidate improvement suggestions

## Run

1. Create a virtual environment.
2. Install `requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Add your Google AI Studio API key.
5. Put PDF, DOCX or TXT resumes into `data/resumes`.
6. Run:

`streamlit run app.py`

Click **Index / Refresh resumes** before searching.

## Example queries

- Find me candidates with React and 3+ years experience
- Compare the top 3 matches side by side
- Why did John rank higher than Jane?
- Add AWS as a must-have and re-rank
- Generate interview questions for the top candidate
- Run the full 3-round screening
