# graph/state.py
#
# PURPOSE:
#   Defines the shared state schema for the LangGraph pipeline.
#   Every node reads from and writes to this state. LangGraph merges
#   each node's return dict into this shared state automatically.
#
# MULTI-AGENT DATA FLOW:
#   Planner  → reads "query",                                    writes "episode_outline"
#   Retrieve → reads "query" + "episode_outline"
#              + "source_filter" + "context_gaps",               writes "documents" + "retrieval_strategy"
#   Writer   → reads "documents" + "episode_outline"
#              + "improvement_suggestions",                       writes "script"
#   Grader   → reads "script",                                   writes "editor_action" + "editor_notes"
#                                                                        + "context_gaps" + "is_acceptable"
#   Editor   → reads "script" + "editor_notes",                  writes "improvement_suggestions"
#   Audio    → reads "script",                                   writes "audio_output"

from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    The shared state for the Blog-to-Podcast LangGraph pipeline.

    Attributes:
        url:                    The blog URL (ingestion reference, not runtime).
        query:                  The user's search query (e.g., "AI Agents").
        source_filter:          Optional source URL/path to scope ChromaDB retrieval.
                                When set, only chunks from that source are searched.
        documents:              Retrieved blog chunks (populated by Retriever Agent).
        episode_outline:        Structured plan produced by the Planner Agent (dict).
        retrieval_strategy:     The Retriever's tool-use summary (tools used, doc counts).
        script:                 The generated podcast script text.
        is_acceptable:          Whether the Editor accepted the script (backward compat).
        script_evaluation:      The Editor's textual feedback (backward compat).
        editor_action:          Triadic routing decision: "accept" | "revise_script"
                                | "request_more_context".
        editor_notes:           Detailed Editor notes for the Writer or Retriever.
        context_gaps:           Specific topics/concepts the Retriever must search for,
                                populated only when editor_action == "request_more_context".
        improvement_suggestions: Writer instructions from the Improvements node.
        generation_count:       Current script generation attempt number.
        max_generations:        Maximum allowed generation attempts.
        audio_output:           File path to the generated MP3 audio.
    """

    url: str
    query: str
    source_filter: Optional[str]           # Phase 1: optional ChromaDB metadata filter
    documents: List[Document]
    episode_outline: Optional[dict]         # Planner Agent output
    retrieval_strategy: Optional[dict]      # Retriever Agent tool-use summary
    script: str
    is_acceptable: bool                     # Backward compat: True iff editor_action == "accept"
    script_evaluation: str                  # Backward compat: same as editor_notes
    editor_action: Optional[str]            # Phase 2: triadic routing ("accept" | "revise_script" | "request_more_context")
    editor_notes: Optional[str]             # Phase 2: detailed Editor feedback
    context_gaps: Optional[str]             # Phase 2: topics for Retriever when request_more_context
    improvement_suggestions: str
    generation_count: int
    max_generations: int
    audio_output: str
