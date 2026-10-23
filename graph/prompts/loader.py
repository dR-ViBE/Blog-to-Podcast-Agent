# graph/prompts/loader.py
#
# PURPOSE:
#   Automated version-aware prompt loader using content hashing (SHA-256).
#
# WHY HASH-BASED VERSIONING:
#   In production systems, assigning version numbers manually (like v1, v2)
#   creates merge conflicts and clashing versions when multiple developers
#   work in parallel.
#   
#   Instead, this loader calculates a unique cryptographic hash of the prompt's
#   text content at runtime. This hash acts as the version ID.
#   - If content changes, the version changes automatically.
#   - Two developers editing the same prompt will get different hashes, eliminating clashing.
#
# HOW IT WORKS:
#   1. Prompts are stored as plain .txt files in graph/prompts/:
#          planner.txt, writer.txt, grader.txt ... (no version suffixes needed)
#   2. load_prompt("planner") reads planner.txt, computes its SHA-256 hash.
#   3. If MLflow tracking is enabled, queries/registers the prompt using its hash version.
#   4. The active hash version is injected into LangSmith metadata for every trace.

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# Absolute path to this directory
_PROMPTS_DIR = Path(__file__).parent

# List of prompt names
_PROMPT_NAMES = ["planner", "writer", "grader", "improvements", "retriever"]


def get_content_hash(text: str) -> str:
    """
    Generates a short SHA-256 hash of the prompt content.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def get_prompt_version(name: str) -> str:
    """
    Returns the active version string for a named prompt.
    Under hash-based versioning, this is the short SHA-256 hash of the prompt content.

    Args:
        name: Prompt name (e.g. "planner", "writer", "grader").

    Returns:
        Short SHA-256 version string (e.g. "3a9f8c12").
    """
    file_path = _PROMPTS_DIR / f"{name}.txt"
    if not file_path.exists():
        # Fallback to local _v1 file if master doesn't exist
        fallback_path = _PROMPTS_DIR / f"{name}_v1.txt"
        if fallback_path.exists():
            file_path = fallback_path
        else:
            logger.warning("Prompt file '%s.txt' not found.", name)
            return "unknown"

    local_text = file_path.read_text(encoding="utf-8").strip()
    return get_content_hash(local_text)


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str | None = None) -> str:
    """
    Load a prompt system message.
    Attempts to fetch from the local MLflow Prompt Registry first (keyed by prompt hash).
    If not found, loads from the local filesystem (.txt file), auto-registers
    it in MLflow for future requests, and caches the result.

    Args:
        name:    Prompt name: "planner" | "writer" | "grader" | "improvements" | "retriever"
        version: Content hash version string. If None, computed automatically.

    Returns:
        The full prompt text as a string.
    """
    # 1. Try loading from local filesystem first to get content
    file_path = _PROMPTS_DIR / f"{name}.txt"
    if not file_path.exists():
        file_path = _PROMPTS_DIR / f"{name}_v1.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prompt file '{name}.txt' not found in {_PROMPTS_DIR}."
        )

    local_text = file_path.read_text(encoding="utf-8").strip()
    computed_version = get_content_hash(local_text)

    if version is None:
        version = computed_version

    # 2. Query MLflow Prompt Registry (SQLite backend)
    # If MLflow is disabled or offline, fallback safely to local filesystem content.
    use_mlflow = os.getenv("USE_MLFLOW_PROMPT_REGISTRY", "false").lower() == "true"
    if use_mlflow:
        run_name = f"{name}_{version}"
        try:
            client = MlflowClient()
            exp_name = "Prompt-Registry"
            experiment = client.get_experiment_by_name(exp_name)
            if not experiment:
                experiment_id = client.create_experiment(exp_name)
            else:
                experiment_id = experiment.experiment_id

            # Search for run representing this prompt name and hash version
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                max_results=1
            )

            if runs:
                run = runs[0]
                local_file = client.download_artifacts(run.info.run_id, "prompt.txt")
                with open(local_file, "r", encoding="utf-8") as f:
                    mlflow_text = f.read().strip()
                logger.info("Loaded prompt from MLflow Registry | name=%s | version=%s", name, version)
                return mlflow_text
            else:
                logger.info("Registering prompt in MLflow | runName=%s", run_name)
                with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                    mlflow.log_text(local_text, "prompt.txt")
                    mlflow.set_tag("prompt_name", name)
                    mlflow.set_tag("prompt_version", version)
                return local_text
        except Exception as mlflow_exc:
            logger.warning("MLflow Prompt Registry lookup failed: %s.", mlflow_exc)

    return local_text


def get_all_active_versions() -> dict:
    """
    Returns a dict of all prompt names and their current content-hash versions.
    Used to inject prompt version metadata into LangSmith traces.

    Returns:
        Dict like: {"planner": "3a9f8c12", "writer": "fd90b1e4", ...}
    """
    return {name: get_prompt_version(name) for name in _PROMPT_NAMES}
