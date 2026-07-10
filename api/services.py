# api/services.py
#
# WHY THIS FILE EXISTS (THE SERVICE LAYER PATTERN):
#   Routes should be "thin" — they only handle HTTP concerns.
#   All business logic (invoking the graph, processing state, handling errors)
#   lives here. This is also the ideal place for LangSmith integration and
#   security guard application.
#
# Phase 1 additions:
#   - PII masking: query is scanned/masked BEFORE entering the graph
#   - Prometheus metrics: request counts, duration, cost, token usage
#   - Cost extraction: llm_cost_usd and total_tokens_used from final state
#   - Source provenance: unique sources extracted from retrieved documents
#   - Prompt version metadata in LangSmith traces

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from api.metrics import METRICS
from api.models import FilterOptions, PodcastResponse
from api.security import pii_guard, check_prompt_injection
from graph.graph import app as langgraph_app
from graph.prompts.loader import get_all_active_versions

logger = logging.getLogger(__name__)

AUDIO_OUTPUT_DIR = Path("outputs/audio")

_APP_NAME = os.getenv("APP_NAME", "blog-to-podcast-agent")
_APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
_ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
_LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "podcast-agent")
_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


@traceable(
    name="run_podcast_agent",
    tags=["langgraph", "podcast-pipeline", _ENVIRONMENT],
    metadata={
        "application": _APP_NAME,
        "version": _APP_VERSION,
        "environment": _ENVIRONMENT,
        "langsmith_project": _LANGCHAIN_PROJECT,
        "model_name": _LLM_MODEL,
        "tts_provider": "elevenlabs",
        "vector_store": "chromadb",
    },
)
def run_podcast_agent(
    query: str,
    max_generations: int = 3,
    source_filter: Optional[str] = None,
    filters: Optional[FilterOptions] = None,
    # PII metadata passed in from the route (already processed before graph invocation)
    pii_was_masked: bool = False,
    pii_entities_found: Optional[List[str]] = None,
) -> PodcastResponse:
    """
    Orchestrates the full Blog-to-Podcast LangGraph pipeline for a given query.

    Phase 1 enhancements:
      - Prometheus metrics recorded for request count, duration, cost, tokens
      - Cost and token data extracted from final graph state
      - PII metadata forwarded into state and response
      - Prompt versions logged to LangSmith metadata

    Args:
        query:            The search query (may have been PII-masked by route layer).
        max_generations:  Max script generation retries.
        source_filter:    Legacy URL prefix filter.
        filters:          Advanced FilterOptions (source_type, source_url_prefix).
        pii_was_masked:   Whether the original query had PII masked.
        pii_entities_found: Entity types that were masked.

    Returns:
        PodcastResponse: A fully populated Pydantic response model.

    Raises:
        RuntimeError: Raised if the LangGraph pipeline fails.
    """
    if pii_entities_found is None:
        pii_entities_found = []

    # ── Track active pipeline runs (Prometheus Gauge) ─────────────────────────
    METRICS.active_pipeline_runs.inc()
    start_time = time.monotonic()

    # ── Resolve metadata filter ───────────────────────────────────────────────
    # Support both legacy source_filter and new FilterOptions
    resolved_source_filter = source_filter
    resolved_source_type_filter = None

    if filters:
        if filters.source_url_prefix:
            resolved_source_filter = filters.source_url_prefix
        if filters.source_type:
            resolved_source_type_filter = filters.source_type

    # ── Build initial LangGraph state ─────────────────────────────────────────
    initial_state: dict = {
        "query": query,
        "max_generations": max_generations,
        "generation_count": 0,
        "source_filter": resolved_source_filter,
        "source_type_filter": resolved_source_type_filter,
        # Cost/token tracking — initialised to zero, accumulated by nodes
        "llm_cost_usd": 0.0,
        "total_tokens_used": 0,
        # PII metadata — set by route layer before graph invocation
        "pii_was_masked": pii_was_masked,
        "pii_entities_found": pii_entities_found,
    }

    logger.info(
        "Graph started | query=%r | max_generations=%d | source_filter=%r | source_type=%r",
        query,
        max_generations,
        resolved_source_filter,
        resolved_source_type_filter,
    )

    # ── Build RunnableConfig with rich LangSmith metadata ────────────────────
    # Include prompt versions so every LangSmith trace shows exactly which
    # prompt file was active for this run — critical for debugging regressions.
    prompt_versions = get_all_active_versions()

    run_config = RunnableConfig(
        run_name=f"Podcast | {query[:60]}",
        tags=[
            "podcast-pipeline",
            _ENVIRONMENT,
            f"max_gen_{max_generations}",
        ],
        metadata={
            "application": _APP_NAME,
            "version": _APP_VERSION,
            "environment": _ENVIRONMENT,
            "model_name": _LLM_MODEL,
            "podcast_topic": query,
            "user_query": query,
            "max_generations": max_generations,
            "source_filter": resolved_source_filter,
            "source_type_filter": resolved_source_type_filter,
            # Prompt versioning metadata — shows in every LangSmith trace
            "prompt_versions": prompt_versions,
            # PII metadata
            "pii_was_masked": pii_was_masked,
            "pii_entities_found": pii_entities_found,
        },
    )

    # ── Invoke the LangGraph app ──────────────────────────────────────────────
    try:
        final_state: dict = langgraph_app.invoke(initial_state, config=run_config)
    except Exception as exc:
        # Record failed request in Prometheus
        METRICS.requests_total.labels(status="failed").inc()
        logger.exception("Graph execution failed | query=%r | error=%s", query, exc)
        raise RuntimeError(f"The LangGraph pipeline encountered an error: {exc}") from exc
    finally:
        # Always record duration (even on failure)
        elapsed = time.monotonic() - start_time
        METRICS.pipeline_duration_seconds.observe(elapsed)
        METRICS.active_pipeline_runs.dec()

    # ── Extract cost & token data from final state ────────────────────────────
    llm_cost_usd: float = final_state.get("llm_cost_usd", 0.0)
    total_tokens_used: int = final_state.get("total_tokens_used", 0)

    # ── Record Prometheus metrics ─────────────────────────────────────────────
    is_accepted = final_state.get("is_acceptable", False)
    generation_count = final_state.get("generation_count", 0)

    # Request-level counter
    audio_output = final_state.get("audio_output")
    status_label = "success" if is_accepted else ("no_audio" if not audio_output else "failed")
    METRICS.requests_total.labels(status=status_label).inc()

    # Generation attempts (total across all retries)
    for _ in range(generation_count):
        result_label = "accepted" if is_accepted else "max_reached"
        METRICS.generation_attempts_total.labels(result=result_label).inc()

    # Script quality metrics
    if is_accepted:
        first_attempt = generation_count == 1
        METRICS.scripts_accepted_total.labels(first_attempt=str(first_attempt)).inc()
        script = final_state.get("script", "")
        if script:
            word_count = len(script.split())
            METRICS.script_word_count.observe(word_count)

    # Cost & token metrics
    if llm_cost_usd > 0:
        METRICS.llm_cost_usd_total.labels(model=_LLM_MODEL).inc(llm_cost_usd)
    if total_tokens_used > 0:
        # Approximate prompt/completion split (60/40 typical for generation pipelines)
        METRICS.tokens_used_total.labels(model=_LLM_MODEL, token_type="prompt").inc(
            int(total_tokens_used * 0.6)
        )
        METRICS.tokens_used_total.labels(model=_LLM_MODEL, token_type="completion").inc(
            int(total_tokens_used * 0.4)
        )

    # Retrieval tool usage
    retrieval_strategy = final_state.get("retrieval_strategy") or {}
    for tool_name in retrieval_strategy.get("tools_used", []):
        METRICS.retrieval_tool_calls_total.labels(tool=tool_name).inc()
    docs_found = retrieval_strategy.get("unique_docs", 0)
    if docs_found > 0:
        METRICS.retrieval_docs_found.observe(docs_found)

    # PII metrics (input)
    for entity_type in pii_entities_found:
        METRICS.pii_detections_total.labels(entity_type=entity_type).inc()

    logger.info(
        "Graph finished | query=%r | generation_count=%d | accepted=%s | "
        "cost_usd=%.6f | tokens=%d | elapsed=%.1fs",
        query,
        generation_count,
        is_accepted,
        llm_cost_usd,
        total_tokens_used,
        elapsed,
    )

    # ── Extract audio path ────────────────────────────────────────────────────
    raw_audio_path: str | None = final_state.get("audio_output")
    audio_filename: str | None = None
    if raw_audio_path:
        audio_filename = Path(raw_audio_path).name
        logger.info("Audio file generated | filename=%s", audio_filename)

    # ── Extract source provenance ──────────────────────────────────────────────
    documents = final_state.get("documents", [])
    sources_used: List[str] = []
    seen_sources = set()
    for doc in documents:
        source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
        if source and source not in seen_sources:
            seen_sources.add(source)
            sources_used.append(source)

    # ── Assemble and return the response ─────────────────────────────────────
    return PodcastResponse(
        status="success",
        script=final_state.get("script"),
        audio_path=audio_filename,
        generation_count=generation_count,
        accepted=is_accepted,
        evaluation=final_state.get("script_evaluation"),
        improvement_suggestions=final_state.get("improvement_suggestions"),
        # Phase 1: cost tracking
        llm_cost_usd=round(llm_cost_usd, 8),
        total_tokens_used=total_tokens_used,
        # Phase 1: PII metadata
        pii_was_masked=pii_was_masked,
        pii_entities_found=pii_entities_found,
        # Phase 1: source provenance
        sources_used=sources_used,
    )
