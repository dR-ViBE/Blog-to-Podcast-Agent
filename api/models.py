# api/models.py
#
# WHY THIS FILE EXISTS:
#   Pydantic models act as the contract between the client and the API.
#   They enforce type safety, validate incoming data automatically, and
#   produce clean, self-documented JSON schemas for API consumers.
#   Keeping models in their own file keeps routes.py and services.py clean.

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union


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
    """

    query: str = Field(
        ...,
        min_length=3,
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
            "Optional: Filter retrieval to chunks from a specific source. "
            "Provide a source URL prefix or exact file path that was used during ingestion. "
            "Example: 'https://lilianweng.github.io' retrieves only chunks from that domain. "
            "When None, all ingested sources are searched."
        ),
        examples=["https://lilianweng.github.io", None],
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


# ---------------------------------------------------------------------------
# HEALTH RESPONSE MODEL
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """
    Simple health check response model.
    Used by GET /health to confirm the service is running.
    """

    status: str = Field(default="healthy", description="Service health status.")
