# graph/chains/script_grader_chain.py
#
# UPGRADE: Editor Richer Routing (3-action schema)
# PROMPT VERSIONING: System prompt loaded from graph/prompts/grader_v{N}.txt
# Controlled by GRADER_PROMPT_VERSION env var (default: v1).
#
# BEFORE (binary):
#   is_acceptable: bool  →  True (go to audio) or False (go to improvements)
#
# AFTER (triadic):
#   action: "accept"              → script is good, go to audio
#   action: "revise_script"       → script has issues, writer needs to fix it
#   action: "request_more_context"→ script is thin because retrieval missed key info

from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from graph.prompts.loader import load_prompt, get_prompt_version

llm = ChatGroq(model="llama-3.1-8b-instant")


class GradeScript(BaseModel):
    """
    The Editor Agent's structured decision on a podcast script.

    Three possible actions:
      - accept               : Script meets all quality criteria → go to audio.
      - revise_script        : Script has fixable issues → send Writer improvement notes.
      - request_more_context : Script is thin because source material was insufficient
                               → send Retriever back to find more content.
    """

    action: Literal["accept", "revise_script", "request_more_context"] = Field(
        description=(
            "The Editor's routing decision. "
            "'accept' if the script meets all criteria. "
            "'revise_script' if the writing quality has fixable issues (bad structure, "
            "wrong tone, formatting violations, too short despite having good source material). "
            "'request_more_context' if the script is thin, vague, or lacks specific details "
            "because the source material retrieved was insufficient — the problem is with "
            "retrieval, not with the writing itself."
        )
    )
    notes: str = Field(
        description=(
            "Detailed notes explaining the decision. "
            "If 'accept': briefly state why the script is good. "
            "If 'revise_script': provide specific, actionable instructions for the writer. "
            "If 'request_more_context': describe exactly what information is missing and "
            "what specific topics the Retriever should search for."
        )
    )
    context_gaps: Optional[str] = Field(
        default=None,
        description=(
            "Only populated when action is 'request_more_context'. "
            "A comma-separated list of specific topics or concepts the Retriever "
            "should search for to fill the content gaps. "
            "Example: 'chain-of-thought prompting examples, few-shot learning benchmarks, "
            "recent GPT-4 prompt engineering research'. "
            "Leave as None for accept or revise_script actions."
        ),
    )


structured_llm_grader = llm.with_structured_output(GradeScript)

# ─── Load Versioned Prompt ───────────────────────────────────────────────────
_GRADER_VERSION = get_prompt_version("grader")
_system = load_prompt("grader", _GRADER_VERSION)

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", _system),
    ("human", "Generated Script:\n\n{script}"),
])

script_grader: RunnableSequence = grader_prompt | structured_llm_grader
