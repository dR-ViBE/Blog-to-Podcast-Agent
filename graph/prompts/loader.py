# graph/prompts/loader.py
#
# PURPOSE:
#   Version-aware prompt loader for all LangChain chain prompts.
#
# WHY PROMPT VERSIONING:
#   In production, prompts are living artefacts — they evolve as you discover
#   edge cases, change tone requirements, or tune LLM behaviour. Without
#   versioning you cannot:
#     - Know which prompt was active when a trace ran (critical for debugging)
#     - Roll back a prompt change that regressed quality
#     - A/B test two prompt versions side by side
#     - Meet audit requirements ("show me the exact prompt that generated this output")
#
# HOW IT WORKS:
#   1. Prompts are stored as plain .txt files in graph/prompts/:
#          planner_v1.txt, planner_v2.txt, writer_v1.txt ...
#   2. The active version for each prompt is controlled by environment variables:
#          PLANNER_PROMPT_VERSION=v1   (default: v1)
#          WRITER_PROMPT_VERSION=v1    (default: v1)
#          etc.
#   3. load_prompt("planner") reads the env var → loads the correct file.
#   4. Falls back to v1 if the requested version file doesn't exist.
#   5. The active version is logged to LangSmith metadata on every run.
#
# TO CREATE A NEW PROMPT VERSION:
#   1. Copy graph/prompts/planner_v1.txt → planner_v2.txt
#   2. Edit planner_v2.txt
#   3. Set PLANNER_PROMPT_VERSION=v2 in .env
#   4. Run RAGAS eval to compare v1 vs v2 quality

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Absolute path to this directory — works regardless of where Python is invoked from
_PROMPTS_DIR = Path(__file__).parent

# Mapping from prompt name → environment variable that controls its version
_VERSION_ENV_VARS: dict = {
    "planner": "PLANNER_PROMPT_VERSION",
    "writer": "WRITER_PROMPT_VERSION",
    "grader": "GRADER_PROMPT_VERSION",
    "improvements": "IMPROVEMENTS_PROMPT_VERSION",
    "retriever": "RETRIEVER_PROMPT_VERSION",
}

_DEFAULT_VERSION = "v1"


def get_prompt_version(name: str) -> str:
    """
    Returns the active version string for a named prompt.

    Reads from the corresponding environment variable.
    Falls back to 'v1' if the env var is not set.

    Args:
        name: Prompt name (e.g. "planner", "writer", "grader").

    Returns:
        Version string (e.g. "v1", "v2").
    """
    env_var = _VERSION_ENV_VARS.get(name)
    if not env_var:
        logger.warning("Unknown prompt name '%s'. Using default version '%s'.", name, _DEFAULT_VERSION)
        return _DEFAULT_VERSION

    version = os.getenv(env_var, _DEFAULT_VERSION).strip()
    return version or _DEFAULT_VERSION


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str | None = None) -> str:
    """
    Load a prompt system message from disk.

    The prompt is cached after the first load — files are read once at startup,
    not on every API request. Cache is keyed by (name, version).

    Args:
        name:    Prompt name: "planner" | "writer" | "grader" | "improvements" | "retriever"
        version: Version string (e.g. "v1"). If None, reads from env var.

    Returns:
        The full prompt text as a string.

    Raises:
        FileNotFoundError: If neither the requested version nor the v1 fallback exists.
    """
    if version is None:
        version = get_prompt_version(name)

    file_path = _PROMPTS_DIR / f"{name}_{version}.txt"

    # Graceful fallback: if requested version doesn't exist, fall back to v1
    if not file_path.exists():
        logger.warning(
            "Prompt file not found: %s. Falling back to v1.",
            file_path,
        )
        file_path = _PROMPTS_DIR / f"{name}_v1.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prompt file '{name}_v1.txt' not found in {_PROMPTS_DIR}. "
            f"Create graph/prompts/{name}_v1.txt to enable prompt versioning."
        )

    text = file_path.read_text(encoding="utf-8").strip()
    logger.info(
        "Loaded prompt | name=%s | version=%s | chars=%d | file=%s",
        name,
        version,
        len(text),
        file_path.name,
    )
    return text


def get_all_active_versions() -> dict:
    """
    Returns a dict of all prompt names and their currently active versions.
    Used to inject prompt version metadata into LangSmith traces.

    Returns:
        Dict like: {"planner": "v1", "writer": "v2", "grader": "v1", ...}
    """
    return {name: get_prompt_version(name) for name in _VERSION_ENV_VARS}
