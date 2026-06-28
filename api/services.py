# api/services.py
#
# WHY THIS FILE EXISTS (THE SERVICE LAYER PATTERN):
#   Routes should be "thin" — they should only handle HTTP concerns:
#   parsing the request, calling the service, and returning the response.
#   All business logic (invoking the graph, processing state, handling
#   domain-specific errors) lives here in the service layer.
#
#   This separation means:
#     1. Logic can be tested independently without spinning up HTTP.
#     2. Routes stay readable and short.
#     3. The graph invocation can be changed/mocked in one place.

import logging
import os
from pathlib import Path

from graph.graph import app as langgraph_app  # The compiled LangGraph StateGraph
from api.models import PodcastResponse

# ---------------------------------------------------------------------------
# Module-level logger
# We use a named logger (not root logger) so log output can be filtered
# and routed independently per module in production log configs.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# The directory where ElevenLabs audio files are saved by generate_audio node.
AUDIO_OUTPUT_DIR = Path("outputs/audio")


def run_podcast_agent(query: str, max_generations: int = 3) -> PodcastResponse:
    """
    Orchestrates the full Blog-to-Podcast LangGraph pipeline for a given query.

    This function:
      1. Constructs the initial LangGraph state dictionary.
      2. Invokes the compiled LangGraph app (which runs the full node pipeline).
      3. Extracts the final state fields and maps them to a PodcastResponse.
      4. Handles and re-raises any exceptions with structured logging.

    Args:
        query:           The search query to retrieve blog chunks from ChromaDB.
        max_generations: Max number of script generation retries before stopping.

    Returns:
        PodcastResponse: A fully populated Pydantic response model.

    Raises:
        RuntimeError: If the LangGraph pipeline fails for any reason.
                      The route layer catches this and returns an HTTP 500.
    """

    # -----------------------------------------------------------------------
    # STEP 1: Build the initial state
    # The LangGraph `retrieve_blog_chunks` node reads `query` from state.
    # `max_generations` controls the retry loop in the `decide_next_step`
    # conditional — without it the graph won't know when to stop retrying.
    # -----------------------------------------------------------------------
    initial_state: dict = {
        "query": query,
        "max_generations": max_generations,
        "generation_count": 0,
    }

    logger.info(
        "Graph started | query=%r | max_generations=%d", query, max_generations
    )

    # -----------------------------------------------------------------------
    # STEP 2: Invoke the LangGraph app
    # `app.invoke()` is a synchronous blocking call that runs the full graph
    # and returns the final merged state after all nodes have executed.
    # -----------------------------------------------------------------------
    try:
        final_state: dict = langgraph_app.invoke(initial_state)
    except Exception as exc:
        # Log the full traceback (exc_info=True) so it appears in server logs.
        logger.exception("Graph execution failed | query=%r | error=%s", query, exc)
        raise RuntimeError(
            f"The LangGraph pipeline encountered an error: {exc}"
        ) from exc

    logger.info(
        "Graph finished | query=%r | generation_count=%d | accepted=%s",
        query,
        final_state.get("generation_count", 0),
        final_state.get("is_acceptable", False),
    )

    # -----------------------------------------------------------------------
    # STEP 3: Extract audio path
    # The `generate_audio` node stores the full OS path in `audio_output`.
    # We convert it to just the filename so the client can fetch it via
    # GET /audio/{filename} without exposing the server's directory layout.
    # -----------------------------------------------------------------------
    raw_audio_path: str | None = final_state.get("audio_output")
    audio_filename: str | None = None

    if raw_audio_path:
        audio_filename = Path(raw_audio_path).name  # e.g. "podcast_abc123.mp3"
        logger.info("Audio file generated | filename=%s", audio_filename)
    else:
        logger.info(
            "No audio file was generated (script may not have been accepted within "
            "max_generations=%d)",
            max_generations,
        )

    # -----------------------------------------------------------------------
    # STEP 4: Assemble and return the response model
    # We use .get() with sensible defaults for every field so a partial graph
    # run (e.g. one that ended at END without audio) still returns valid JSON.
    # -----------------------------------------------------------------------
    return PodcastResponse(
        status="success",
        script=final_state.get("script"),
        audio_path=audio_filename,
        generation_count=final_state.get("generation_count", 0),
        accepted=final_state.get("is_acceptable", False),
        evaluation=final_state.get("script_evaluation"),
        improvement_suggestions=final_state.get("improvement_suggestions"),
    )
