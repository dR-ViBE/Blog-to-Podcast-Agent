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

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.models import HealthResponse, PodcastRequest, PodcastResponse
from api.services import run_podcast_agent

# ---------------------------------------------------------------------------
# Module-level logger — mirrors the logger in services.py for easy log tracing
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# APIRouter
# We use a router (not decorators directly on `app`) so that routes.py stays
# completely decoupled from the FastAPI app instance defined in main.py.
# This allows independent testing and future route versioning (e.g. /v2/).
# ---------------------------------------------------------------------------
router = APIRouter()

# Directory where ElevenLabs audio files are saved by the graph
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
        "via ElevenLabs. Returns the final script, evaluation, and audio path."
    ),
    status_code=200,
    tags=["Podcast"],
)
def generate_podcast(request: PodcastRequest) -> PodcastResponse:
    """
    POST /podcast

    Orchestrates the full Blog-to-Podcast pipeline for a given query.

    Args:
        request (PodcastRequest): Validated request body with `query` and
                                  optional `max_generations`.

    Returns:
        PodcastResponse: Structured JSON containing the script, audio path,
                         grading result, and generation metadata.

    Raises:
        HTTPException 422: Raised automatically by FastAPI if request body
                           fails Pydantic validation (e.g. query too short).
        HTTPException 500: Raised if the LangGraph pipeline fails internally.
    """
    # Log the incoming request — useful for debugging and audit trails
    logger.info(
        "Incoming request | endpoint=POST /podcast | query=%r | max_generations=%d",
        request.query,
        request.max_generations,
    )

    # Delegate all graph logic to the service layer
    try:
        response: PodcastResponse = run_podcast_agent(
            query=request.query,
            max_generations=request.max_generations,
            source_filter=request.source_filter,
        )
    except RuntimeError as exc:
        # RuntimeError is raised by services.py when the graph itself fails.
        # We convert it to a 500 HTTP response with a clear message.
        logger.error(
            "Pipeline error | query=%r | error=%s", request.query, exc
        )
        raise HTTPException(
            status_code=500,
            detail=f"Podcast pipeline failed: {exc}",
        )
    except Exception as exc:
        # Catch-all for any unexpected error not covered by services.py
        logger.exception(
            "Unexpected error | query=%r | error=%s", request.query, exc
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred. Check server logs.",
        )

    logger.info(
        "Request completed | endpoint=POST /podcast | query=%r | accepted=%s",
        request.query,
        response.accepted,
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

    FastAPI's `FileResponse` handles:
      - Setting the correct `Content-Type: audio/mpeg` header.
      - Efficient chunked streaming (no full file loaded into memory).
      - HTTP Range request support for audio seeking in browsers.

    Args:
        filename: The MP3 filename returned by POST /podcast (e.g.
                  "podcast_abc123def456.mp3"). Must not contain path
                  traversal characters — FastAPI path parameter parsing
                  handles this automatically.

    Returns:
        FileResponse: Streams the MP3 file to the client.

    Raises:
        HTTPException 400: If the filename contains path traversal attempts.
        HTTPException 404: If the file does not exist in the audio directory.
    """
    # Sanitise the filename: strip any directory components to prevent
    # path-traversal attacks (e.g. filename = "../../etc/passwd")
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
