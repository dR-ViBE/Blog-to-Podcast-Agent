# graph/chains/outline_model.py
#
# PURPOSE:
#   Defines the Pydantic models that represent the Planner Agent's output.
#   These models are used as the structured output schema for ChatGroq —
#   the LLM is forced to return JSON that validates against these models.
#
# WHY PYDANTIC MODELS FOR LLM OUTPUT:
#   LangChain's `.with_structured_output(MyModel)` tells the LLM to return
#   JSON matching the model's schema. If the LLM returns invalid JSON,
#   LangChain retries automatically. This guarantees the planner always
#   produces a well-structured outline that the writer can follow.
#
# ARCHITECTURE ROLE:
#   These models define the CONTRACT between the Planner Agent and the
#   Writer Agent. The Planner produces an EpisodeOutline, which is stored
#   in GraphState. The Writer reads it and follows the structure.
#
#   Planner Agent ──produces──▶ EpisodeOutline ──consumed by──▶ Writer Agent

from typing import List
from pydantic import BaseModel, Field


class TalkingPoint(BaseModel):
    """
    A single section of the podcast episode body.

    The Planner breaks the episode into 3-5 talking points, each covering
    one key idea from the retrieved blog content. The Writer uses these
    to structure the body of the script in the correct order.

    Attributes:
        topic:                      The section heading / theme for this talking point.
                                    Example: "What are AI Agents?"
        key_insight:                The core idea the writer must convey in this section.
                                    Example: "AI Agents are autonomous systems that can
                                    plan, reason, and take actions to achieve goals."
        estimated_duration_seconds: How many seconds this section should take when
                                    spoken aloud. Helps the writer calibrate length.
                                    Example: 45 (means ~45 seconds of speaking time)
    """

    topic: str = Field(
        description="The theme or heading for this section of the podcast body."
    )
    key_insight: str = Field(
        description=(
            "The single most important idea the writer must convey in this section. "
            "Should be 1-2 sentences of clear, plain-English explanation."
        )
    )
    estimated_duration_seconds: int = Field(
        default=45,
        ge=15,    # Minimum 15 seconds per section (prevents trivial points)
        le=120,   # Maximum 2 minutes per section (prevents monolithic blocks)
        description="Estimated speaking duration for this section in seconds.",
    )


class EpisodeOutline(BaseModel):
    """
    The complete episode plan produced by the Planner Agent.

    This is the structured output that ChatGroq is forced to generate.
    Every field provides specific guidance to the Writer Agent, ensuring
    the podcast script has consistent quality, structure, and pacing.

    The outline is stored in GraphState as a dict (via .model_dump())
    because LangGraph's TypedDict state requires JSON-serializable values.

    Attributes:
        episode_title:                     A catchy, engaging title for the episode.
        hook:                              The very first sentence — must grab attention.
        key_talking_points:                Ordered list of body sections (3-5 points).
        target_audience:                   Who this episode is for (helps tone calibration).
        tone_guidance:                     Specific tone instructions for the writer.
        total_estimated_duration_seconds:  Target total duration in seconds (180-300).
        sign_off_suggestion:               How to end the episode.
    """

    episode_title: str = Field(
        description=(
            "A catchy, engaging title for the podcast episode. "
            "Should be 5-10 words, suitable for a podcast directory listing. "
            "Example: 'AI Agents: Your New Digital Coworkers'"
        )
    )

    hook: str = Field(
        description=(
            "The opening line of the podcast — the very first thing the host says. "
            "Must grab attention immediately. Should be a surprising fact, "
            "a provocative question, or a bold statement. 1-2 sentences max."
        )
    )

    key_talking_points: List[TalkingPoint] = Field(
        min_length=3,   # At least 3 sections for substance
        max_length=5,   # At most 5 sections to stay within 3-5 minutes
        description=(
            "An ordered list of 3-5 talking points that form the body of the episode. "
            "Each point covers one key idea from the source material. "
            "The writer MUST follow this order."
        )
    )

    target_audience: str = Field(
        description=(
            "A one-sentence description of who this episode is for. "
            "Example: 'Tech-curious professionals who want to understand AI "
            "without diving into code.'"
        )
    )

    tone_guidance: str = Field(
        description=(
            "Specific tone and style instructions for the writer. "
            "Example: 'Keep it conversational and upbeat. Use analogies "
            "from everyday life. Avoid jargon unless you explain it immediately.'"
        )
    )

    total_estimated_duration_seconds: int = Field(
        ge=180,   # Minimum 3 minutes
        le=300,   # Maximum 5 minutes
        description=(
            "Target total speaking duration for the entire episode in seconds. "
            "Must be between 180 (3 min) and 300 (5 min). "
            "The sum of talking point durations + intro + outro should match this."
        )
    )

    sign_off_suggestion: str = Field(
        description=(
            "A suggestion for how the host should close the episode. "
            "Example: 'Wrap up by challenging listeners to try building "
            "their own AI agent this weekend.'"
        )
    )
