# graph/chains/planner_chain.py
#
# PURPOSE:
#   Defines the Planner Agent's LangChain chain.
#   This chain takes the user query AND retrieved documents and produces a
#   structured EpisodeOutline via ChatGroq's structured output mode.
#
# CHANGE (Architecture Fix #2 — Retrieve-first):
#   BEFORE: Planner received only {query} and planned from LLM general knowledge.
#   AFTER:  Planner receives {query} + {context} (retrieved docs) and plans
#           EXCLUSIVELY from the source material. This enforces document-centric
#           podcast generation — the outline is grounded in the user's content.
#
# PROMPT VERSIONING:
#   The system prompt is loaded from graph/prompts/planner_v{N}.txt
#   controlled by the PLANNER_PROMPT_VERSION environment variable (default: v1).
#   This allows prompt A/B testing and safe rollback without code changes.
#
# CHAIN FLOW:
#   planner_prompt (ChatPromptTemplate)
#       │
#       ▼
#   ChatGroq.with_structured_output(EpisodeOutline)
#       │
#       ▼
#   EpisodeOutline (validated Pydantic object)

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from graph.chains.outline_model import EpisodeOutline
from graph.prompts.loader import load_prompt, get_prompt_version

# ─── LLM Configuration (Two-Tier Fallback) ──────────────────────────────────
# Tier 1 (Primary)  : Groq openai/gpt-oss-20b — fast, tool-calling, structured output.
# Tier 2 (Fallback) : Groq openai/gpt-oss-120b — more capable Groq model.
# Tier 3 (Local)    : Ollama phi3 — free, runs locally, zero external dependency.
#                     Activates if Groq is unreachable. Requires Ollama/Docker running.
_primary_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, max_tokens=2500)
_groq_fallback = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, max_tokens=2500)
_local_fallback = ChatOllama(model="phi3:latest", temperature=0.0)

llm = _primary_llm.with_fallbacks(
    [_groq_fallback, _local_fallback],
    exceptions_to_handle=(Exception,),
)
structured_planner_llm = llm.with_structured_output(EpisodeOutline)

# ─── Load Versioned Prompt ───────────────────────────────────────────────────
# Reads PLANNER_PROMPT_VERSION from env (defaults to "v1").
# The active version is logged at import time for observability.
_PLANNER_VERSION = get_prompt_version("planner")
_system = load_prompt("planner", _PLANNER_VERSION)

# ─── Prompt Template ─────────────────────────────────────────────────────────
# {context} = retrieved document chunks (the actual source material to plan from)
# {query}   = user's search query (used as a focus/angle, not as the content source)
planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system),
        (
            "human",
            "USER QUERY: {query}\n\n"
            "SOURCE DOCUMENTS (plan ONLY from content present here):\n"
            "---\n"
            "{context}\n"
            "---",
        ),
    ]
)

# ─── Assembled Chain ─────────────────────────────────────────────────────────
# planner_prompt → structured_planner_llm → EpisodeOutline
planner_chain = planner_prompt | structured_planner_llm

