# Architecture

User
 |
 v
Streamlit conversational UI
 |
 v
LangGraph matching agent
 |
 +-- Parse intent
 |
 +-- Parse JD
 |
 +-- Extract requirements
 |
 +-- Hybrid RAG search
 |    +-- Chroma vector retrieval
 |    +-- resume metadata
 |    +-- objective experience filtering
 |
 +-- Rank candidates
 |
 +-- Deep candidate analysis
 |
 +-- Generate report
 |
 +-- Human feedback
 |    +-- accept -> END
 |    +-- refinement -> merge requirements -> re-rank
 |
 +-- Compare
 +-- Why ranked
 +-- Interview questions
 +-- 3-round screening

## Separation of concerns

agent/
    LangGraph state and orchestration

tools/
    Agent-callable recruiting tools

services/
    Gemini generation and deterministic matching

rag/
    Chunking, embeddings and vector search

filesystem/
    PDF/DOCX/TXT parsing and file operations

data/
    Resumes, jobs and Chroma persistence

The LLM handles extraction, reasoning, explanations and generation.
Deterministic code handles retrieval, metadata constraints and numeric ranking.
