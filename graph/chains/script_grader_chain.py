# graph/chains/script_grader_chain.py
#
# UPGRADE: Editor Richer Routing (3-action schema)
#
# BEFORE (binary):
#   is_acceptable: bool  →  True (go to audio) or False (go to improvements)
#
# AFTER (triadic):
#   action: "accept"              → script is good, go to audio
#   action: "revise_script"       → script has issues, writer needs to fix it
#   action: "request_more_context"→ script is thin because retrieval missed key info,
#                                   send the Retriever back to fetch more
#
# WHY THIS MATTERS:
#   The old binary grader couldn't distinguish BETWEEN two failure modes:
#     1. Bad writing  → Writer needs better instructions (revise_script)
#     2. Thin content → Retriever needs to find better sources (request_more_context)
#   Mixing these caused the Writer to get improvement suggestions for problems
#   it couldn't fix (missing information it was never given).
#
#   Now the Editor diagnoses the ROOT CAUSE and routes accordingly.
#   This is real agentic decision-making: the LLM output drives graph routing.

from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

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
        )
    )


structured_llm_grader = llm.with_structured_output(GradeScript)

system = """You are the Podcast Editor Agent — a Senior Podcast Producer with 20 years of experience.
Your job is to quality-control a monologue script for \"The Insight Loop\" podcast.

**GRADING CRITERIA:**

1. **Length:** The script must be substantial for a 3-5 minute episode.
   - Roughly 450 to 800 words.
   - REJECT if under 400 words — but first diagnose WHY it is short.

2. **Structure:**
   - Must have a clear Intro (Hook + Welcome).
   - Must have a Body (Main content covering the key talking points).
   - Must have an Outro (Summary + Sign-off).

3. **Format (CRITICAL):**
   - Clean spoken word only.
   - REJECT if it contains Markdown headers (## Section, **Host:**).
   - REJECT if it contains non-verbal cues ([laughs], [music plays]).
   - REJECT if it is a bullet list instead of full prose.

4. **Content Depth:**
   - The script must contain SPECIFIC details, examples, and insights — not just general statements.
   - If the script is vague, generic, or repeats the same basic idea without depth, it fails.

**DECISION RULES — choose your action carefully:**

Choose **"accept"** if the script meets all criteria above.

Choose **"revise_script"** if:
- The script has BAD WRITING (wrong tone, markdown formatting, missing structure, poor flow).
- The content is there but poorly written or structured.
- The writer has enough material but didn't use it well.

Choose **"request_more_context"** if:
- The script is thin or vague even though the writing itself is decent.
- The writer seems to have been working with insufficient source material.
- The script repeats the same generic statements without specific details, examples, or data.
- Key talking points from the outline are barely covered or skipped.
- In this case, the problem is NOT the writer — it's that the Retriever didn't find enough content.

**IMPORTANT:** Do not choose "request_more_context" just because the script is short.
Only choose it when the writing quality is acceptable but the content is clearly thin due to lack of source material.
"""

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Generated Script:\n\n{script}"),
])

script_grader: RunnableSequence = grader_prompt | structured_llm_grader
