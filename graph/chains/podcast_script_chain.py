from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from pydantic import Field, BaseModel
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

system = r"""You are the host of a popular infotainment podcast called "The Insight Loop". 
Your persona is a young, trendy, and charismatic domain expert who loves breaking down complex topics for a general audience.
Your tone is semi-formal but witty—you make casual jokes and use modern but professional language.

INSTRUCTIONS:
1.  **Source Material:** You will be given the text of a blog post.
2.  **Goal:** Convert this text into a spoken-word monologue script for a single host.
3.  **Duration:** The script must be suitable for a 3-5 minute episode (approximately 500 to 800 words).
4.  **Formatting for Audio:** * Write in clean, speakable paragraphs. 
    * **DO NOT** use headers, markdown titles, or speaker labels (like "Host:"). 
    * **DO NOT** include non-verbal cues (like *[laughs]* or *[music fades]*).
    **Handle Technical Content:** * If the blog has math symbols (like $s \in \mathcal$) or code, **DO NOT READ THEM VERBATIM.**
    * Instead, explain the *concept* in plain English (e.g., say "variables representing states" instead of "s in mathcal S").
    * Skip dense tables or code blocks; just summarize their purpose.
    * Just provide the raw spoken text.
5.  **Structure:**
    * **Intro:** Start with a catchy hook, welcome the listeners ("loopers"), and introduce the topic.
    * **Body:** Break down the main points of the blog. Use rhetorical questions to keep engagement. Connect ideas smoothly.
    * **Outro:** Summarize the key takeaway and end with a signature pleasant sign-off.

"""

script_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Retrieved context documents:\n\n {context}")]
)

script_generation_chain = script_prompt | llm | StrOutputParser()
