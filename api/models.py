# api/models.py
#
# WHY THIS FILE EXISTS:
#   Pydantic models act as the contract between the client and the API.
#   They enforce type safety, validate incoming data automatically, and
#   produce clean, self-documented JSON schemas for API consumers.
#   Keeping models in their own file keeps routes.py and services.py clean.

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# FILTER OPTIONS (Phase 1: enhanced metadata filtering)
# ---------------------------------------------------------------------------


class FilterOptions(BaseModel):
    """
    Advanced retrieval filter options for the ChromaDB vector store.

    These filters are applied to ChromaDB metadata fields set during ingestion.
    Filters can be combined — e.g. "only search PDF files from this domain".

    The ingestion pipeline sets the following metadata fields:
        source:       The URL or absolute file path of the original document.
        source_type:  One of "url", "pdf", "text", "directory".
        ingested_at:  ISO 8601 timestamp when the document was ingested.

    Attributes:
        source_url_prefix: Restrict retrieval to documents whose `source` field
                           contains this string. Useful for domain-scoping.
                           Example: "https://lilianweng.github.io" → only that blog.
        source_type:       Restrict retrieval to documents of a specific type.
                           Example: "pdf" → only search ingested PDF documents.
    """

    source_url_prefix: Optional[str] = Field(
        default=None,
        description=(
            "Restrict retrieval to sources whose URL/path contains this substring. "
            "Example: 'https://lilianweng.github.io' retrieves only chunks from that domain."
        ),
        examples=["https://lilianweng.github.io", "/docs/research/"],
    )
    source_type: Optional[Literal["url", "pdf", "text", "directory"]] = Field(
        default=None,
        description=(
            "Restrict retrieval to a specific source type. "
            "One of: 'url' (web pages), 'pdf' (PDF files), 'text' (plain text/markdown), "
            "'directory' (directory-ingested files). "
            "Leave None to search all content types."
        ),
        examples=["pdf", "url", None],
    )


# ---------------------------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------------------------


class PodcastRequest(BaseModel):
    """
    Defines the shape of the incoming POST /podcast request body.

    Attributes:
        query:           The search query used to retrieve relevant blog chunks
                         from the ChromaDB vector store (e.g. "AI Agents").
        max_generations: How many times the LangGraph loop is allowed to
                         regenerate the script before it gives up and returns
                         the best attempt. Defaults to 3.
        source_filter:   Legacy field: URL prefix to scope ChromaDB retrieval.
                         Prefer using `filters.source_url_prefix` instead.
        filters:         Advanced metadata filters for ChromaDB retrieval.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The topic or search query to generate a podcast episode about.",
        examples=["AI Agents", "Prompt Engineering techniques"],
    )
    max_generations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of script generation attempts before the agent stops.",
    )
    source_filter: Optional[str] = Field(
        default=None,
        description=(
            "Legacy: Filter retrieval to chunks from a specific source URL/path. "
            "Prefer using 'filters.source_url_prefix' for new integrations."
        ),
        examples=["https://lilianweng.github.io", None],
    )
    filters: Optional[FilterOptions] = Field(
        default=None,
        description=(
            "Advanced retrieval filters. Combine source URL prefix and source type "
            "to precisely scope what content is retrieved. "
            "Example: {source_type: 'pdf', source_url_prefix: '/docs/'} searches only "
            "PDFs from the /docs/ path."
        ),
    )


# ---------------------------------------------------------------------------
# RESPONSE MODEL
# ---------------------------------------------------------------------------


class PodcastResponse(BaseModel):
    """
    Defines the shape of the JSON response returned by POST /podcast.

    WHY RESPONSE MODELS MATTER:
      - They guarantee the API always returns the same shape, even if the
        internal graph state changes in the future.
      - FastAPI uses this model to auto-generate OpenAPI documentation.
      - Optional fields handle cases where audio was not generated (e.g.,
        max retries reached without an acceptable script).

    Attributes:
        status:                  "success" or "failed" string indicator.
        script:                  The final generated podcast script text.
        audio_path:              Relative path to the saved .mp3 file, or None
                                 if audio generation was not reached.
        generation_count:        How many generation attempts were made.
        accepted:                Whether the script passed the quality grader.
        evaluation:              The grader's textual feedback on the script.
        improvement_suggestions: The editor's improvement bullet points (if any).
        llm_cost_usd:            Estimated LLM cost for this run in USD.
        total_tokens_used:       Estimated total tokens consumed (prompt + completion).
        pii_was_masked:          True if PII was detected and masked in the input query.
        pii_entities_found:      List of PII entity types that were detected/masked.
        sources_used:            Unique source URLs/paths that contributed documents.
    """

    status: str = Field(description="Indicates whether the pipeline succeeded.")
    script: Optional[str] = Field(default=None, description="The final podcast script.")
    audio_path: Optional[str] = Field(
        default=None,
        description="Relative path to the generated MP3 file inside outputs/audio/.",
    )
    generation_count: int = Field(
        default=0, description="Number of script generation attempts made."
    )
    accepted: bool = Field(
        default=False, description="True if the grader accepted the final script."
    )
    evaluation: Optional[str] = Field(
        default=None, description="Grader's reason for acceptance or rejection."
    )
    improvement_suggestions: Optional[str] = Field(
        default=None,
        description="Editor's improvement suggestions from the last failed grade cycle.",
    )
    # ── Cost & Performance Metrics (Phase 1) ─────────────────────────────────
    llm_cost_usd: float = Field(
        default=0.0,
        description=(
            "Estimated total LLM cost for this run in USD. "
            "Based on character-count estimation (~4 chars/token). "
            "Accuracy: ~92-95% vs exact token counts. "
            "Good range: $0.0001-$0.005 per run on Groq llama-3.1-8b-instant."
        ),
    )
    total_tokens_used: int = Field(
        default=0,
        description=(
            "Estimated total tokens consumed (prompt + completion) across all LLM calls "
            "in the pipeline. Includes planner, retriever, writer, grader, and improvements nodes."
        ),
    )
    # ── Security / Privacy Metadata (Phase 1) ────────────────────────────────
    pii_was_masked: bool = Field(
        default=False,
        description="True if PII was detected and masked in the input query before processing.",
    )
    pii_entities_found: List[str] = Field(
        default_factory=list,
        description=(
            "PII entity types detected in the input query (e.g. ['PERSON', 'EMAIL_ADDRESS']). "
            "Empty list if no PII was detected or if presidio is not installed."
        ),
    )
    # ── Retrieval Provenance (Phase 1) ────────────────────────────────────────
    sources_used: List[str] = Field(
        default_factory=list,
        description=(
            "Unique source URLs or file paths of documents that contributed to the final script. "
            "Useful for attribution and audit trails."
        ),
    )


# ---------------------------------------------------------------------------
# HEALTH RESPONSE MODEL
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """
    Simple health check response model.
    Used by GET /health to confirm the service is running.
    """

    status: str = Field(default="healthy", description="Service health status.")
