# graph/state.py
#
# PURPOSE:
#   Defines the shared state schema for the LangGraph pipeline.
#   Every node reads from and writes to this state. LangGraph merges
#   each node's return dict into this shared state automatically.
#
# WHY TypedDict:
#   LangGraph uses TypedDict (not a Pydantic model) for state because:
#     - It's a plain dict at runtime — fast and JSON-serializable
#     - Type annotations are for developer tooling, not runtime validation
#     - LangGraph nodes return partial dicts that are merged into state
#
# MULTI-AGENT DATA FLOW:
#   Retrieve → writes "documents"
#   Planner  → reads "documents" + "query", writes "episode_outline"
#   Writer   → reads "documents" + "episode_outline" + "improvement_suggestions", writes "script"
#   Grader   → reads "script", writes "is_acceptable" + "script_evaluation"
#   Editor   → reads "script" + "script_evaluation", writes "improvement_suggestions"
#   Audio    → reads "script", writes "audio_output"

from typing import Dict, List, Optional
from langchain_core.documents import Document


class GraphState(Dict):
    """
    The shared state for the Blog-to-Podcast LangGraph pipeline.

    Each field represents a piece of data that flows between agents (nodes).
    Nodes only write the fields they own — LangGraph merges the partial
    dict into the full state automatically.

    Attributes:
        url:                        The blog URL (set during ingestion, not runtime).
        query:                      The user's search query (e.g., "AI Agents").
        documents:                  Retrieved blog chunks from ChromaDB.
        episode_outline:            Structured plan produced by the Planner Agent.
                                    Stored as a dict (from EpisodeOutline.model_dump()).
        script:                     The generated podcast script text.
        is_acceptable:              Whether the grader accepted the script.
        script_evaluation:          The grader's textual reasoning.
        improvement_suggestions:    Editor's feedback for rejected scripts.
        generation_count:           Current script generation attempt number.
        max_generations:            Maximum allowed generation attempts.
        audio_output:               File path to the generated MP3 audio.
    """

    url: str
    query: str
    documents: List[Document]
    episode_outline: Optional[dict]       # NEW: Planner Agent output
    script: str
    is_acceptable: bool
    script_evaluation: str
    improvement_suggestions: str
    editor_action: Optional[str]          # NEW: Editor Agent decision (accept/revision)
    editor_notes: Optional[str]           # NEW: Editor Agent detailed feedback
    generation_count: int
    max_generations: int
    audio_output: str
