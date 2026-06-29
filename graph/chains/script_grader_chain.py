from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")


from typing import Literal

class GradeScript(BaseModel):
    """Editor decision and detailed feedback on the podcast script."""
    action: Literal["accept", "revision"] = Field(
        description="The editor's decision: 'accept' if the script is perfect, 'revision' if changes are needed."
    )
    notes: str = Field(
        description="Detailed revision notes for the writer if action is 'revision'. If 'accept', briefly state why it is good."
    )

structured_llm_grader = llm.with_structured_output(GradeScript)

system = """You are the Podcast Editor Agent. 
You think like a Senior Podcast Producer.
Your job is to quality-control a monologue script generated for the "The Insight Loop" podcast.

**GRADING CRITERIA:**

1.  **Length:** The script should be substantial enough for a 3-5 minute episode. 
    * *Check:* Does it have roughly 450 to 800 words? 
    * If it is extremely short (under 400 words), reject it.

2.  **Structure:** * Must have a clear **Intro** (Hook + Welcome).
    * Must have a **Body** (Main content).
    * Must have an **Outro** (Summary + Sign-off).

3.  **Format (CRITICAL):**
    * The text must be **clean spoken word**.
    * **REJECT** if it contains Markdown headers (like ## Intro, **Host:**).
    * **REJECT** if it contains non-verbal cues (like *[laughs]*, *[music plays]*).
    * **REJECT** if it is just a summary list; it must be full sentences.

4.  **Tone:**
    * Must be engaging, infotainment style (not a boring lecture).

**OUTPUT INSTRUCTION:**
Return 'accept' if the script meets all criteria.
Return 'revision' and provide DETAILED revision notes if the script violates ANY of these criteria.
"""

grader_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Generated Script:\n\n{script}")])

script_grader: RunnableSequence = grader_prompt | structured_llm_grader
