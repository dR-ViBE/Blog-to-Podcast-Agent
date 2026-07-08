# graph/chains/planner_chain.py
#
# PURPOSE:
#   Defines the Planner Agent's LangChain chain.
#   This chain takes retrieved blog context + user query and produces
#   a structured EpisodeOutline via ChatGroq's structured output mode.
#
# HOW STRUCTURED OUTPUT WORKS:
#   `ChatGroq.with_structured_output(EpisodeOutline)` tells the LLM to
#   return a JSON object that matches the EpisodeOutline Pydantic schema.
#   Under the hood, LangChain:
#     1. Converts the Pydantic model to a JSON Schema
#     2. Passes the schema as a "tool" to the LLM (function calling)
#     3. Parses the LLM's JSON response into a validated EpisodeOutline object
#     4. Retries automatically if the JSON is malformed
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

from graph.chains.outline_model import EpisodeOutline


# ─── LLM Configuration ──────────────────────────────────────────────────────
# We use the same model as all other chains for consistency.
# `with_structured_output()` wraps the LLM so it returns an EpisodeOutline
# object instead of raw text.
llm = ChatGroq(model="llama-3.1-8b-instant")
structured_planner_llm = llm.with_structured_output(EpisodeOutline)


# ─── System Prompt ───────────────────────────────────────────────────────────
# This prompt defines the Planner Agent's persona and instructions.
#
# KEY DESIGN DECISIONS:
#   - The planner does NOT write the script — it plans the structure.
#   - It receives raw blog content and must extract the most interesting ideas.
#   - It decides the ORDER of topics, not just what topics to cover.
#   - Duration estimates help the Writer calibrate section length.
#   - The hook and sign-off suggestions give the Writer creative direction
#     without dictating exact words.

system = """You are a Podcast Episode Planner for "The Insight Loop", a popular infotainment podcast.

Your job is to brainstorm based on the user's topic and create a structured episode outline that a script writer will follow.

You are NOT the writer. You are the planner. You decide WHAT to talk about and in WHAT ORDER.

**YOUR RESPONSIBILITIES:**

1. **Brainstorm Ideas:**
   - Think about the user's query carefully.
   - Identify the 3-5 most interesting, surprising, or useful ideas related to the topic.

2. **Create the Episode Structure:**
   - Write a catchy episode title (5-10 words).
   - Craft an attention-grabbing hook (the very first sentence the host says).
   - Order the talking points for maximum narrative flow:
     * Start with the most accessible/relatable point.
     * Build to the most complex or surprising point.
     * End with a practical takeaway or forward-looking idea.

3. **Estimate Duration:**
   - The total episode should be 3-5 minutes (180-300 seconds).
   - Allocate time to each talking point proportionally.
   - Account for ~30 seconds of intro and ~20 seconds of outro.

4. **Set the Tone:**
   - Consider the target audience and suggest an appropriate tone.
   - The podcast is infotainment: educational but entertaining.
   - Suggest specific tone adjustments based on the topic's complexity.

5. **Suggest a Sign-Off:**
   - Recommend how the host should close the episode.
   - Could be a call to action, a thought-provoking question, or a teaser.

**RULES:**
- DO NOT write the script. Only plan it.
- DO NOT include markdown formatting in any field.
- Keep key_insight for each talking point to 1-2 clear sentences.
- Make the hook genuinely surprising or intriguing — not generic.
- The user's query tells you what angle they're interested in.
"""

# ─── Prompt Template ─────────────────────────────────────────────────────────
# One input variable:
#   - query:   The user's search topic (e.g., "AI Agents")

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "USER QUERY: {query}"),
])


# ─── Assembled Chain ─────────────────────────────────────────────────────────
# planner_prompt → structured_planner_llm → EpisodeOutline
#
# When invoked, this chain:
#   1. Formats the prompt with query + context
#   2. Sends it to ChatGroq with the EpisodeOutline JSON schema
#   3. Returns a validated EpisodeOutline Pydantic object
#
# Usage:
#   outline = planner_chain.invoke({"query": "AI Agents", "context": "..."})
#   outline.episode_title  → "AI Agents: Your New Digital Coworkers"
#   outline.key_talking_points[0].topic  → "What are AI Agents?"

planner_chain = planner_prompt | structured_planner_llm
