# graph/chains/improvements_chain.py
#
# PROMPT VERSIONING: System prompt loaded from graph/prompts/improvements_v{N}.txt
# Controlled by IMPROVEMENTS_PROMPT_VERSION env var (default: v1).

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from graph.prompts.loader import load_prompt, get_prompt_version

llm = ChatGroq(model="llama-3.1-8b-instant", max_tokens=2500).with_fallbacks([ChatGroq(model="llama-3.3-70b-versatile", max_tokens=2500)])

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
