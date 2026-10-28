# graph/chains/improvements_chain.py
#
# PROMPT VERSIONING: System prompt loaded from graph/prompts/improvements_v{N}.txt
# Controlled by IMPROVEMENTS_PROMPT_VERSION env var (default: v1).

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from graph.prompts.loader import load_prompt, get_prompt_version

# ─── LLM Configuration (Two-Tier Fallback) ──────────────────────────────────
# Tier 1 (Primary)  : Groq llama-3.1-8b-instant — fast, cheap.
# Tier 2 (Fallback) : Groq llama-3.3-70b-versatile — more capable Groq model.
# Tier 3 (Local)    : Ollama phi3 — free local fallback if Groq is unreachable.
_primary_llm = ChatGroq(model="llama-3.1-8b-instant", max_tokens=2500)
_groq_fallback = ChatGroq(model="llama-3.3-70b-versatile", max_tokens=2500)
_local_fallback = ChatOllama(model="phi3:latest")

llm = _primary_llm.with_fallbacks(
    [_groq_fallback, _local_fallback],
    exceptions_to_handle=(Exception,),
)

# ─── Load Versioned Prompt ───────────────────────────────────────────────────
_IMPROVEMENTS_VERSION = get_prompt_version("improvements")
_system = load_prompt("improvements", _IMPROVEMENTS_VERSION)

improvement_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system),
        ("human", "REJECTED SCRIPT: {script}\n\nREJECTION REASON:{script_evaluation}"),
    ]
)

suggest_improvements_chain = improvement_prompt | llm | StrOutputParser()
