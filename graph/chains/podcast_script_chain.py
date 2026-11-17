# graph/chains/podcast_script_chain.py
#
# PROMPT VERSIONING:
#   System prompt loaded from graph/prompts/writer_v{N}.txt
#   Controlled by WRITER_PROMPT_VERSION env var (default: v1).

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from graph.prompts.loader import get_prompt_version, load_prompt

# ─── LLM Configuration (Two-Tier Fallback) ──────────────────────────────────
# Tier 1 (Primary)  : Groq llama-3.1-8b-instant — fast, cheap.
# Tier 2 (Fallback) : Groq llama-3.3-70b-versatile — more capable Groq model.
# Tier 3 (Local)    : Ollama phi3 — free local fallback if Groq is unreachable.
_primary_llm = ChatGroq(model="openai/gpt-oss-20b", max_tokens=2500)
_groq_fallback = ChatGroq(model="openai/gpt-oss-120b", max_tokens=2500)
_local_fallback = ChatOllama(model="phi3:latest")

llm = _primary_llm.with_fallbacks(
    [_groq_fallback, _local_fallback],
    exceptions_to_handle=(Exception,),
)

# ─── Load Versioned Prompt ───────────────────────────────────────────────────
_WRITER_VERSION = get_prompt_version("writer")
_system = load_prompt("writer", _WRITER_VERSION)

script_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system),
        (
            "human",
            "EPISODE OUTLINE (follow this structure exactly):\n\n{episode_outline}\n\n"
            "RETRIEVED BLOG CONTENT (your source material):\n\n{context}\n\n"
            "EDITOR IMPROVEMENT SUGGESTIONS (if any — apply all of them):\n\n{improvement_suggestions}",
        ),
    ]
)

script_generation_chain = script_prompt | llm | StrOutputParser()
