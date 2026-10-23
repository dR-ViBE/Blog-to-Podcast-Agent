# api/metrics.py
#
# PURPOSE:
#   Centralised Prometheus metrics registry for the Blog-to-Podcast Agent.
#
# WHY PROMETHEUS:
#   Prometheus is the industry-standard metrics system for containerised services.
#   It uses a pull model — Prometheus scrapes our /metrics endpoint periodically.
#   From there, Grafana can visualise every metric defined here as dashboards.
#
# METRIC TYPES USED:
#   - Counter:   Monotonically increasing value (resets on restart).
#                Use for: request counts, error counts, cost accumulation.
#                Reading: "rate(metric[5m])" gives events per second over 5 min.
#
#   - Histogram: Tracks value distribution across configurable buckets.
#                Use for: latency, token counts, cost per request.
#                Reading: "histogram_quantile(0.95, rate(...))" gives p95 latency.
#
#   - Gauge:     Current point-in-time value (can go up and down).
#                Use for: queue depth, active connections, cache size.
#
# HOW TO USE:
#   from api.metrics import METRICS
#   METRICS.requests_total.labels(status="success").inc()
#   METRICS.pipeline_duration.observe(elapsed_seconds)

import logging
from dataclasses import dataclass

from prometheus_client import Counter, Histogram, Gauge, REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# METRIC DEFINITIONS
# ---------------------------------------------------------------------------

@dataclass
class PodcastMetrics:
    """
    All Prometheus metrics for the Blog-to-Podcast Agent.

    Each metric has:
      - A descriptive name (prefixed with `podcast_`)
      - A help string explaining what it measures
      - Optional label names for dimensional slicing

    Labels let you filter/group metrics in Prometheus queries.
    Example: `podcast_requests_total{status="success"}` vs `{status="failed"}`
    """

    # ── Request-level counters ────────────────────────────────────────────────

    requests_total: Counter
    """
    Total POST /podcast requests, labelled by final status.
    Labels: status = "success" | "failed" | "no_audio"
    Normal: Mostly "success". High "failed" rate = pipeline or LLM issue.
    """

    # ── Pipeline performance ──────────────────────────────────────────────────

    pipeline_duration_seconds: Histogram
    """
    End-to-end pipeline latency from request received to response sent.
    Buckets: 5s, 15s, 30s, 60s, 90s, 120s, 180s
    Normal range: 20-60s (LLM + TTS is inherently slow).
    Alert if: p95 > 120s — indicates LLM timeout or overload.
    """

    generation_attempts_total: Counter
    """
    Script generation loop iterations (each = one LLM call to the Writer).
    Labels: result = "accepted" | "rejected_revise" | "rejected_context" | "max_reached"
    Normal: Most runs complete in 1-2 attempts.
    Alert if: "max_reached" rate > 20% — Writer struggling, check prompts.
    """

    # ── LLM Cost & Tokens ─────────────────────────────────────────────────────

    llm_cost_usd_total: Counter
    """
    Cumulative LLM cost in USD, labelled by model.
    Labels: model = "llama-3.1-8b-instant" | "other"
    Use rate(podcast_llm_cost_usd_total[1h]) to see hourly spend.
    Good: < $0.01 per run on Groq free tier (llama-3.1-8b is very cheap).
    """

    tokens_used_total: Counter
    """
    Total tokens consumed (prompt + completion combined), labelled by model and type.
    Labels: model, token_type = "prompt" | "completion"
    Use to track token efficiency: high prompt/completion ratio = prompt bloat.
    """

    # ── Script quality ────────────────────────────────────────────────────────

    scripts_accepted_total: Counter
    """
    Scripts accepted by the Editor Agent on first attempt (no revisions needed).
    Good: > 70% acceptance rate on first attempt.
    Bad: < 40% = Writer or Grader prompt needs tuning.
    """

    script_word_count: Histogram
    """
    Word count of accepted podcast scripts.
    Buckets: 200, 350, 450, 550, 650, 750, 900
    Target range: 450-800 words (3-5 minute episode at ~150 wpm).
    Alert if: median < 350 = scripts too short.
    """

    # ── Security events ───────────────────────────────────────────────────────

    injection_attempts_total: Counter
    """
    Prompt injection attempts blocked by the security guard.
    Labels: reason = "pattern_match" | "length_exceeded"
    Normal: Near zero. Any sustained rate = active attack, consider IP blocking.
    """

    pii_detections_total: Counter
    """
    PII entities detected in queries, labelled by entity type.
    Labels: entity_type = "PERSON" | "EMAIL_ADDRESS" | "PHONE_NUMBER" | etc.
    Note: High PERSON count is expected for research queries.
    Alert if: CREDIT_CARD, US_SSN, IBAN_CODE > 0 — unexpected sensitive data.
    """

    pii_output_detections_total: Counter
    """
    PII detected in generated scripts (output scanning).
    This should be near zero — PII in output means ingested documents contained PII.
    Alert if: any count > 0 — review ingested content.
    """

    # ── Retrieval pipeline ────────────────────────────────────────────────────

    retrieval_tool_calls_total: Counter
    """
    Tool calls made by the Retriever Agent, labelled by tool name.
    Labels: tool = "search_vectorstore" | "search_web"
    Normal: Mostly vectorstore. High web_search rate = local corpus thin.
    """

    retrieval_docs_found: Histogram
    """
    Number of unique documents found per retrieval run (after dedup, before rerank).
    Buckets: 1, 2, 4, 6, 8, 10, 15
    Alert if: median < 2 = corpus too sparse, ingest more content.
    """

    # ── Ingestion ─────────────────────────────────────────────────────────────

    ingestion_chunks_total: Counter
    """
    Total document chunks ingested into ChromaDB, labelled by source_type.
    Labels: source_type = "url" | "pdf" | "text" | "directory"
    Useful for understanding corpus composition over time.
    """

    ingestion_requests_total: Counter
    """
    Total POST /ingest requests, labelled by source_type and status.
    Labels: source_type, status = "success" | "failed"
    """

    # ── System health ─────────────────────────────────────────────────────────

    active_pipeline_runs: Gauge
    """
    Currently executing podcast pipeline runs (in-flight requests).
    Normal: 0-3 (pipeline is synchronous and slow).
    Alert if: sustained > 5 = requests backing up, server under stress.
    """


def _build_metrics() -> PodcastMetrics:
    """Creates and registers all Prometheus metrics. Called once at module load."""

    # Histogram bucket boundaries chosen to match expected latency distributions
    DURATION_BUCKETS = [5.0, 15.0, 30.0, 60.0, 90.0, 120.0, 180.0, 300.0]
    WORD_COUNT_BUCKETS = [100, 200, 350, 450, 550, 650, 750, 900, 1200]
    DOC_COUNT_BUCKETS = [1, 2, 4, 6, 8, 10, 15, 20]

    return PodcastMetrics(
        # Request-level
        requests_total=Counter(
            "podcast_requests_total",
            "Total podcast generation requests by final status",
            ["status"],
        ),
        # Pipeline performance
        pipeline_duration_seconds=Histogram(
            "podcast_pipeline_duration_seconds",
            "End-to-end pipeline execution time in seconds",
            buckets=DURATION_BUCKETS,
        ),
        generation_attempts_total=Counter(
            "podcast_generation_attempts_total",
            "Script generation loop iterations by result",
            ["result"],
        ),
        # LLM cost & tokens
        llm_cost_usd_total=Counter(
            "podcast_llm_cost_usd_total",
            "Cumulative estimated LLM cost in USD by model",
            ["model"],
        ),
        tokens_used_total=Counter(
            "podcast_tokens_used_total",
            "Total tokens consumed by model and type (prompt/completion)",
            ["model", "token_type"],
        ),
        # Script quality
        scripts_accepted_total=Counter(
            "podcast_scripts_accepted_total",
            "Scripts accepted by Editor on first attempt vs after revisions",
            ["first_attempt"],
        ),
        script_word_count=Histogram(
            "podcast_script_word_count",
            "Word count distribution of accepted podcast scripts",
            buckets=WORD_COUNT_BUCKETS,
        ),
        # Security
        injection_attempts_total=Counter(
            "podcast_injection_attempts_total",
            "Prompt injection attempts blocked by security guard",
            ["reason"],
        ),
        pii_detections_total=Counter(
            "podcast_pii_detections_total",
            "PII entities detected in input queries by entity type",
            ["entity_type"],
        ),
        pii_output_detections_total=Counter(
            "podcast_pii_output_detections_total",
            "PII entities detected in generated scripts by entity type",
            ["entity_type"],
        ),
        # Retrieval
        retrieval_tool_calls_total=Counter(
            "podcast_retrieval_tool_calls_total",
            "Tool calls made by Retriever Agent by tool name",
            ["tool"],
        ),
        retrieval_docs_found=Histogram(
            "podcast_retrieval_docs_found",
            "Unique documents found per retrieval run after deduplication",
            buckets=DOC_COUNT_BUCKETS,
        ),
        # Ingestion
        ingestion_chunks_total=Counter(
            "podcast_ingestion_chunks_total",
            "Document chunks ingested into ChromaDB by source type",
            ["source_type"],
        ),
        ingestion_requests_total=Counter(
            "podcast_ingestion_requests_total",
            "Total ingestion requests by source type and status",
            ["source_type", "status"],
        ),
        # System health
        active_pipeline_runs=Gauge(
            "podcast_active_pipeline_runs",
            "Currently executing podcast pipeline runs",
        ),
    )


# ---------------------------------------------------------------------------
# MODULE-LEVEL SINGLETON
# ---------------------------------------------------------------------------
# All metrics are registered with Prometheus once at import time.
# Import this object wherever you need to record metrics:
#   from api.metrics import METRICS
#   METRICS.requests_total.labels(status="success").inc()

METRICS = _build_metrics()
logger.info("Prometheus metrics registry initialised with %d metrics.", 14)
