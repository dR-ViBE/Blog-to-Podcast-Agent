# api/routes.py
#
# WHY ROUTES STAY "THIN":
#   A route function is responsible for exactly three things:
#     1. Accepting and validating the HTTP request (Pydantic does this).
#     2. Calling the appropriate service function.
#     3. Returning the HTTP response.
#
#   No graph logic, no file I/O, no business rules belong here.
#   This makes routes trivially easy to read and test.
#
# Phase 1 additions:
#   - POST /podcast: prompt injection check + PII masking before service call
#   - POST /ingest: Prometheus counter for ingestion events
#   - GET /metrics: documented (endpoint mounted in main.py)

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, File, UploadFile, Form

from fastapi.responses import FileResponse

from api.metrics import METRICS
from api.models import FilterOptions, HealthResponse, PodcastRequest, PodcastResponse
from api.security import check_prompt_injection, pii_guard
from api.services import run_podcast_agent
from ingestion.loaders import ingest_source

logger = logging.getLogger(__name__)

router = APIRouter()

AUDIO_OUTPUT_DIR = Path("outputs/audio")


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns a simple `{status: healthy}` payload. "
    "Used by load balancers and uptime monitors.",
    tags=["Utility"],
)
def health_check() -> HealthResponse:
    """
    Lightweight liveness probe.

    No database or graph dependencies are checked here — this endpoint
    should always return 200 as long as the Python process is alive.
    """
    return HealthResponse(status="healthy")


# ---------------------------------------------------------------------------
# GENERATE PODCAST
# ---------------------------------------------------------------------------


@router.post(
    "/podcast",
    response_model=PodcastResponse,
    summary="Generate Podcast Episode",
    description=(
        "Accepts a query string, retrieves relevant blog content from the "
        "ChromaDB vector store, generates a podcast script using the LangGraph "
        "pipeline, grades and iteratively improves it, then synthesises audio "
        "via ElevenLabs. Returns the final script, evaluation, audio path, "
        "LLM cost estimate, PII detection results, and source provenance."
    ),
    status_code=200,
    tags=["Podcast"],
)
def generate_podcast(request: PodcastRequest) -> PodcastResponse:
    """
    POST /podcast

    Phase 1 security additions (applied before any LLM call):
      1. Prompt Injection Check: blocks queries matching known attack patterns.
      2. PII Masking: detects and masks PII in the query (PERSON, EMAIL, etc.)
         before sending it to the LLM pipeline.

    Args:
        request (PodcastRequest): Validated request body.

    Returns:
        PodcastResponse: Script, audio path, cost, PII metadata, and sources.

    Raises:
        HTTPException 400: Prompt injection detected, or query too long.
        HTTPException 422: Pydantic validation failure.
        HTTPException 500: LangGraph pipeline internal failure.
    """
    logger.info(
        "Incoming request | endpoint=POST /podcast | query=%r | max_generations=%d",
        request.query,
        request.max_generations,
    )

    # ── Security Layer 1: Prompt Injection Check ──────────────────────────────
    # Runs BEFORE any LLM call. Raises HTTP 400 if injection detected.
    # This is the first line of defence against OWASP LLM Top 10 #1 risk.
    try:
        check_prompt_injection(request.query)
    except HTTPException:
        # Re-raise as-is (already has correct status code and message)
        METRICS.injection_attempts_total.labels(reason="pattern_match").inc()
        raise

    # ── Security Layer 2: PII Detection & Masking ────────────────────────────
    # Scan and mask PII in the query before it reaches the LLM.
    # We record what was masked so the response can inform the caller.
    masked_query = request.query
    pii_was_masked = False
    pii_entities_found = []

    if pii_guard.is_available:
        masked_query, pii_entities_found = pii_guard.mask_text(request.query)
        pii_was_masked = (masked_query != request.query)

        if pii_was_masked:
            logger.info(
                "Query PII masked | entity_types=%s | original_len=%d | masked_len=%d",
                pii_entities_found,
                len(request.query),
                len(masked_query),
            )

    # ── Call the Service Layer ────────────────────────────────────────────────
    try:
        response: PodcastResponse = run_podcast_agent(
            query=masked_query,
            max_generations=request.max_generations,
            source_filter=request.source_filter,
            filters=request.filters,
            pii_was_masked=pii_was_masked,
            pii_entities_found=pii_entities_found,
        )
    except RuntimeError as exc:
        logger.error("Pipeline error | query=%r | error=%s", request.query, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Podcast pipeline failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error | query=%r | error=%s", request.query, exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred. Check server logs.",
        )

    # ── PII: Scan generated script output (do not mask — just warn) ───────────
    if pii_guard.is_available and response.script:
        output_pii = pii_guard.scan_text(response.script)
        if output_pii:
            for entity_type in output_pii:
                METRICS.pii_output_detections_total.labels(entity_type=entity_type).inc()
            logger.warning(
                "PII detected in generated script output | entity_types=%s",
                output_pii,
            )

    logger.info(
        "Request completed | endpoint=POST /podcast | query=%r | accepted=%s | cost_usd=%.6f",
        request.query,
        response.accepted,
        response.llm_cost_usd,
    )
    return response


# ---------------------------------------------------------------------------
# SERVE AUDIO FILE
# ---------------------------------------------------------------------------


@router.get(
    "/audio/{filename}",
    summary="Download Generated Audio",
    description=(
        "Serves a previously generated podcast MP3 file by filename. "
        "The filename is returned by POST /podcast in the `audio_path` field."
    ),
    tags=["Podcast"],
    responses={
        200: {"content": {"audio/mpeg": {}}, "description": "MP3 audio file stream."},
        404: {"description": "Audio file not found."},
    },
)
def serve_audio(filename: str) -> FileResponse:
    """
    GET /audio/{filename}

    Streams an MP3 file from the `outputs/audio/` directory.
    Path traversal prevention: strips any directory components from filename.
    """
    # Sanitise: strip directory components to prevent path traversal (../../etc/passwd)
    safe_filename = Path(filename).name
    if safe_filename != filename:
        logger.warning(
            "Path traversal attempt detected | requested=%r | safe=%r",
            filename,
            safe_filename,
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Directory traversal is not permitted.",
        )

    audio_file_path = AUDIO_OUTPUT_DIR / safe_filename
    logger.info("Audio file requested | filename=%s", safe_filename)

    if not audio_file_path.exists():
        logger.warning("Audio file not found | path=%s", audio_file_path)
        raise HTTPException(
            status_code=404,
            detail=f"Audio file '{safe_filename}' not found. "
            "Ensure the podcast was generated successfully.",
        )

    return FileResponse(
        path=str(audio_file_path),
        media_type="audio/mpeg",
        filename=safe_filename,
    )


# ---------------------------------------------------------------------------
# INGEST DOCUMENT OR URL
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    summary="Ingest Document or URL",
    description=(
        "Upload a PDF/text file or provide a URL to ingest into ChromaDB. "
        "Ingested documents become searchable by the Retriever Agent. "
        "Supports URL crawling, PDF parsing, and plain text/markdown files."
    ),
    tags=["Podcast"],
)
def ingest_data(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
):
    """
    POST /ingest

    Accepts either a file upload or a URL string.
    Calls the unified ingest_source() function which handles type detection,
    chunking, embedding, and ChromaDB storage.
    """
    if not file and not url:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file or a URL to ingest.",
        )

    upload_dir = Path("outputs/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Determine source_type for Prometheus label
    source_type = "url" if url else (
        "pdf" if file and file.filename and file.filename.lower().endswith(".pdf")
        else "text"
    )

    if file:
        file_path = upload_dir / file.filename
        try:
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            chunks = ingest_source(str(file_path))

            METRICS.ingestion_chunks_total.labels(source_type=source_type).inc(chunks)
            METRICS.ingestion_requests_total.labels(
                source_type=source_type, status="success"
            ).inc()

            return {
                "status": "success",
                "message": f"Successfully ingested {chunks} chunks from file: {file.filename}",
                "chunks": chunks,
                "source": str(file_path.resolve()),
                "source_type": source_type,
            }
        except Exception as exc:
            METRICS.ingestion_requests_total.labels(
                source_type=source_type, status="failed"
            ).inc()
            logger.exception("Ingestion failed for file: %s", file.filename)
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {exc}",
            )

    if url:
        try:
            chunks = ingest_source(url)

            METRICS.ingestion_chunks_total.labels(source_type="url").inc(chunks)
            METRICS.ingestion_requests_total.labels(
                source_type="url", status="success"
            ).inc()

            return {
                "status": "success",
                "message": f"Successfully ingested {chunks} chunks from URL: {url}",
                "chunks": chunks,
                "source": url,
                "source_type": "url",
            }
        except Exception as exc:
            METRICS.ingestion_requests_total.labels(
                source_type="url", status="failed"
            ).inc()
            logger.exception("Ingestion failed for URL: %s", url)
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {exc}",
            )
