# graph/chains/podcast_script_chain.py
#
# PROMPT VERSIONING:
#   System prompt loaded from graph/prompts/writer_v{N}.txt
#   Controlled by WRITER_PROMPT_VERSION env var (default: v1).

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from graph.prompts.loader import load_prompt, get_prompt_version

llm = ChatGroq(model="llama-3.1-8b-instant")

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
