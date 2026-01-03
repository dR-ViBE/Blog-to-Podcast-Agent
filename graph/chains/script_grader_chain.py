from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")


class GradeScript(BaseModel):
    """Binary score for the generated script is acceptable or not,Reason why it is not acceptable"""
    is_acceptable: bool = Field(
        description="script meets said podcast quality conditions,'true' or 'false'")
    reason: str = Field(
        description="Give clear ,concise reason why the script is not acceptable")


structured_llm_grader = llm.with_structured_output(GradeScript)

system = """You are a Senior Podcast Producer and Editor. 
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
Return a JSON object with `is_acceptable` (boolean) and `reason` (string).
"""

grader_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Generated Script:\n\n{script}")])

script_grader: RunnableSequence = grader_prompt | structured_llm_grader
