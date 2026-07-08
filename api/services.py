# api/services.py
#
# WHY THIS FILE EXISTS (THE SERVICE LAYER PATTERN):
#   Routes should be "thin" — they only handle HTTP concerns.
#   All business logic (invoking the graph, processing state, handling errors)
#   lives here. This is also the ideal place for LangSmith integration because:
#     - It's the single entry point into the LangGraph pipeline.
#     - All custom metadata about the request (query, max_generations, etc.)
#       is available here before the graph runs.
#     - Wrapping the graph call here means every trace in LangSmith carries
#       the same rich context without touching individual node files.

import logging
import os
from pathlib import Path

# LangSmith's @traceable decorator wraps any Python function and creates a
# parent trace in LangSmith. All LangChain/LangGraph calls made inside the
# function automatically become child runs of this parent trace.
# This gives you a single "root" trace per user request in LangSmith.
from langsmith import traceable

# RunnableConfig is LangChain's standard way to pass runtime configuration
# to any chain or graph invocation. It carries:
#   - metadata  : a dict of key/value pairs visible in LangSmith traces
#   - tags       : a list of searchable string labels on the trace
#   - run_name   : the display name of the run in LangSmith UI
from langchain_core.runnables import RunnableConfig

from graph.graph import app as langgraph_app  # The compiled LangGraph StateGraph
from api.models import PodcastResponse

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# The directory where ElevenLabs audio files are saved by generate_audio node.
AUDIO_OUTPUT_DIR = Path("outputs/audio")

# ---------------------------------------------------------------------------
# READ APPLICATION METADATA FROM ENVIRONMENT
#
# We read these once at module load time (not inside the function) so they
# are fetched from the environment only once, not on every API call.
# These are set in .env and loaded by load_dotenv() in api/main.py.
# ---------------------------------------------------------------------------
_APP_NAME = os.getenv("APP_NAME", "blog-to-podcast-agent")
_APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
_ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
_LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "podcast-agent")


# ---------------------------------------------------------------------------
# SERVICE FUNCTION
# ---------------------------------------------------------------------------

@traceable(
    # `name` is the display label for this function's trace in LangSmith UI.
    name="run_podcast_agent",

    # `tags` appear as coloured labels on the trace in LangSmith.
    # Useful for filtering runs (e.g. filter by "langgraph" or "production").
    tags=["langgraph", "podcast-pipeline", _ENVIRONMENT],

    # `metadata` is a flat dict of key/value pairs stored with every trace.
    # These are static values that don't change per request.
    # Per-request dynamic metadata (query, max_generations) is added below
    # inside the function via RunnableConfig.
    metadata={
        "application": _APP_NAME,
        "version": _APP_VERSION,
        "environment": _ENVIRONMENT,
        "langsmith_project": _LANGCHAIN_PROJECT,
        "model_name": "llama-3.1-8b-instant",  # Groq model used by all chains
        "tts_provider": "elevenlabs",
        "vector_store": "chromadb",
    },
)
def run_podcast_agent(query: str, max_generations: int = 3, source_filter: str = None) -> PodcastResponse:
    """
    Orchestrates the full Blog-to-Podcast LangGraph pipeline for a given query.

    LangSmith Integration:
      The @traceable decorator (above) creates a parent trace for this entire
      function call. The langgraph_app.invoke() call below creates a child trace
      (the LangGraph run), and within it each node creates its own child trace,
      and within each node every chain call creates yet another child trace.

      The full trace hierarchy in LangSmith will look like:

        run_podcast_agent                       ← @traceable (this function)
        └── Blog-to-Podcast Pipeline            ← LangGraph graph run
            ├── retrieve                        ← LangGraph node
            ├── generate_script                 ← LangGraph node
            │   ├── ChatPromptTemplate          ← chain step
            │   ├── ChatGroq                    ← LLM call (tokens, latency)
            │   └── StrOutputParser             ← chain step
            ├── grade_script                    ← LangGraph node
            │   ├── ChatPromptTemplate
            │   └── ChatGroq (structured)       ← structured output LLM call
            ├── suggest_improvements            ← LangGraph node (if needed)
            │   ├── ChatPromptTemplate
            │   ├── ChatGroq
            │   └── StrOutputParser
            └── generate_audio                  ← LangGraph node

    Args:
        query:           The search query to retrieve blog chunks from ChromaDB.
        max_generations: Max number of script generation retries before stopping.

    Returns:
        PodcastResponse: A fully populated Pydantic response model.

    Raises:
        RuntimeError: Raised if the LangGraph pipeline fails. The route layer
                      catches this and returns an HTTP 500 to the client.
    """

    # -----------------------------------------------------------------------
    # STEP 1: Build the initial LangGraph state
    # -----------------------------------------------------------------------
    initial_state: dict = {
        "query": query,
        "max_generations": max_generations,
        "generation_count": 0,
        "source_filter": source_filter,  # None means search all sources
    }

    logger.info(
        "Graph started | query=%r | max_generations=%d", query, max_generations
    )

    # -----------------------------------------------------------------------
    # STEP 2: Build RunnableConfig with rich LangSmith metadata
    #
    # WHY RunnableConfig?
    #   When you pass a RunnableConfig to langgraph_app.invoke(), LangGraph
    #   automatically propagates it to EVERY node in the graph and every
    #   chain call within those nodes. This means you write metadata ONCE
    #   here and it flows through the entire pipeline automatically.
    #
    # WHAT APPEARS IN LANGSMITH?
    #   - metadata    → visible in the "Metadata" tab of each trace
    #   - tags        → shown as colour-coded labels; filterable in the UI
    #   - run_name    → the display name of the LangGraph run in the UI
    # -----------------------------------------------------------------------
    run_config = RunnableConfig(
        # run_name becomes the title of the LangGraph run in LangSmith.
        # Using the query makes it easy to identify runs at a glance.
        run_name=f"Podcast | {query[:60]}",

        # tags are searchable labels — good for filtering in LangSmith UI.
        # e.g. you can filter all runs tagged "max_gen_3" or "development".
        tags=[
            "podcast-pipeline",
            _ENVIRONMENT,
            f"max_gen_{max_generations}",
        ],

        # metadata is the richest part — every key/value here appears in
        # LangSmith under the "Metadata" tab and is fully searchable.
        metadata={
            # ── Static application metadata ──────────────────────────────
            "application": _APP_NAME,
            "version": _APP_VERSION,
            "environment": _ENVIRONMENT,
            "model_name": "llama-3.1-8b-instant",

            # ── Per-request dynamic metadata ─────────────────────────────
            # These values change with every request, giving you full
            # observability into exactly what each run was processing.
            "podcast_topic": query,           # What the user asked for
            "user_query": query,              # Same as above, aliased for clarity
            "max_generations": max_generations,  # Retry budget for this run
        },
    )

    # -----------------------------------------------------------------------
    # STEP 3: Invoke the LangGraph app with the RunnableConfig
    #
    # By passing `run_config` here, we attach all the metadata above to
    # the graph run AND all child runs (nodes, chains, LLM calls) inside it.
    # -----------------------------------------------------------------------
    try:
        final_state: dict = langgraph_app.invoke(initial_state, config=run_config)

    except Exception as exc:
        logger.exception("Graph execution failed | query=%r | error=%s", query, exc)
        # RuntimeError is caught by routes.py and returned as HTTP 500
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
    # STEP 4: Extract audio path
    # Convert the full OS path stored by generate_audio node to just the
    # filename, so the client can fetch it via GET /audio/{filename}.
    # -----------------------------------------------------------------------
    raw_audio_path: str | None = final_state.get("audio_output")
    audio_filename: str | None = None

    if raw_audio_path:
        audio_filename = Path(raw_audio_path).name
        logger.info("Audio file generated | filename=%s", audio_filename)
    else:
        logger.info(
            "No audio generated (script not accepted within max_generations=%d)",
            max_generations,
        )

    # -----------------------------------------------------------------------
    # STEP 5: Assemble and return the response model
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
