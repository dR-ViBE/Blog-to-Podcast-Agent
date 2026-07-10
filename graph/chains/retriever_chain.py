# graph/chains/retriever_chain.py
#
# PROMPT VERSIONING: System prompt loaded from graph/prompts/retriever_v{N}.txt
# Controlled by RETRIEVER_PROMPT_VERSION env var (default: v1).

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from graph.prompts.loader import load_prompt, get_prompt_version


class RetrievalStrategy(BaseModel):
    """Structured strategy for multi-query retrieval based on the planner's outline."""

    main_topic: str = Field(description="The core topic of the episode.")
    important_concepts: str = Field(description="Key concepts that must be explained.")
    supporting_concepts: str = Field(description="Secondary ideas to add depth.")
    examples_needed: str = Field(description="Types of examples or case studies required.")
    recent_developments: str = Field(description="Latest trends or news related to the topic.")
    definitions_needed: str = Field(description="Jargon or technical terms that need definition.")
    queries: list[str] = Field(
        description="A list of 3-5 distinct search queries to retrieve documents covering all the dimensions above."
    )


llm = ChatGroq(model="llama-3.1-8b-instant")
structured_retriever_llm = llm.with_structured_output(RetrievalStrategy)

# ─── Load Versioned Prompt ───────────────────────────────────────────────────
_RETRIEVER_VERSION = get_prompt_version("retriever")
_system = load_prompt("retriever", _RETRIEVER_VERSION)

retriever_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system),
        ("human", "EPISODE OUTLINE:\n\n{outline}"),
    ]
)

retriever_chain = retriever_prompt | structured_retriever_llm
