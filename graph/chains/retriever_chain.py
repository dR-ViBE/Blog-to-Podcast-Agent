from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


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

system = """You are an Enterprise Retrieval Agent. 
Your job is to read an episode outline provided by the Planner Agent and develop a comprehensive search strategy.

**YOUR INSTRUCTIONS:**
1. Analyze the Episode Outline.
2. Break down the information needs into 6 dimensions:
   - Main Topic
   - Important Concepts
   - Supporting Concepts
   - Examples Needed
   - Recent Developments
   - Definitions Needed
3. Based on these dimensions, formulate 3 to 5 distinct semantic search queries. 
   These queries will be run against a vector database to fetch the exact documents the Writer Agent will need.

Make the queries targeted and specific. Do not use generic terms.
"""

retriever_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "EPISODE OUTLINE:\n\n{outline}"),
    ]
)

retriever_chain = retriever_prompt | structured_retriever_llm
