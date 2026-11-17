# graph/checkpointer.py
#
# PURPOSE:
#   Provides a LangGraph-compatible checkpointer so the pipeline can survive
#   crashes mid-run and state can be resumed from the last successful node.
#
# WHY CHECKPOINTING?
#   Without a checkpointer, if the server crashes mid-pipeline (e.g. during
#   `generate_script`, which takes 20-40s), all prior work is lost and the
#   user must start over. With a checkpointer attached to the compiled graph,
#   every node's output is snapshotted before execution moves to the next.
#   On replay (same thread_id), LangGraph loads the saved state and resumes
#   from the last successfully completed node.
#
# HOW TO USE:
#   Import `get_checkpointer()` in graph.py and pass it to workflow.compile():
#       from graph.checkpointer import get_checkpointer
#       app = workflow.compile(checkpointer=get_checkpointer())
#
#   In services.py, set a deterministic thread_id in the RunnableConfig:
#       config = RunnableConfig(configurable={"thread_id": thread_id})
#
# CHECKPOINTER SELECTION:
#   - SqliteSaver (preferred): Requires `langgraph-checkpoint-sqlite` package.
#     Persists checkpoints to disk → survives server restarts.
#     NOTE: langgraph-checkpoint-sqlite 3.x requires langgraph-checkpoint>=4.x,
#           which conflicts with langgraph 1.0.5. Upgrade langgraph to 2.x first
#           if you need true disk persistence.
#   - MemorySaver (current default): Ships with langgraph itself (no extra install).
#     Stores checkpoints in RAM → survives node crashes within a single run,
#     but state is lost on server restart. Sufficient for in-process recovery.
#
# CHECKPOINT DB LOCATION (SqliteSaver only):
#   Default: outputs/checkpoints/graph_state.db
#   Override via CHECKPOINT_DB_PATH env var (e.g. for Docker volume mounting).

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RESOLVE THE CHECKPOINT DB PATH (used if SqliteSaver becomes available)
# ---------------------------------------------------------------------------
_CHECKPOINT_DB_PATH = os.environ.get(
    "CHECKPOINT_DB_PATH",
    "outputs/checkpoints/graph_state.db",
)

# Ensure parent directory exists so the file path is ready when needed
Path(_CHECKPOINT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CREATE THE SINGLETON CHECKPOINTER
# ---------------------------------------------------------------------------
# Try SqliteSaver first (persistent across restarts). If not available due to
# package version constraints, fall back to MemorySaver (in-process only).

_checkpointer = None
_CHECKPOINTER_AVAILABLE = False
_CHECKPOINTER_TYPE = "none"

try:
    # SqliteSaver requires langgraph-checkpoint-sqlite installed separately.
    # On langgraph 1.0.5 this means langgraph-checkpoint-sqlite < 2.0.
    import contextlib

    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver

    _exit_stack = contextlib.ExitStack()
    _checkpointer = _exit_stack.enter_context(_SqliteSaver.from_conn_string(_CHECKPOINT_DB_PATH))
    _CHECKPOINTER_AVAILABLE = True
    _CHECKPOINTER_TYPE = "SqliteSaver"
    logger.info(
        "LangGraph SqliteSaver checkpointer ACTIVE | db=%s | checkpoints persist across restarts",
        Path(_CHECKPOINT_DB_PATH).resolve(),
    )

except (ImportError, ModuleNotFoundError):
    # SqliteSaver package not installed or version incompatible with this
    # langgraph version. Use MemorySaver as the in-process checkpointer.
    try:
        from langgraph.checkpoint.memory import MemorySaver as _MemorySaver

        _checkpointer = _MemorySaver()
        _CHECKPOINTER_TYPE = "MemorySaver"
        logger.info(
            "LangGraph MemorySaver checkpointer ACTIVE | "
            "checkpoints survive node failures within a single process run. "
            "To enable disk persistence, install a compatible "
            "langgraph-checkpoint-sqlite version."
        )
    except Exception as mem_exc:
        logger.warning("MemorySaver initialisation failed (%s). Checkpointing disabled.", mem_exc)

except Exception as _e:
    # Catch any other errors (e.g. DB locked, permission error) and fall back
    logger.warning("SqliteSaver initialisation failed (%s). Falling back to MemorySaver.", _e)
    try:
        from langgraph.checkpoint.memory import MemorySaver as _MemorySaver

        _checkpointer = _MemorySaver()
        _CHECKPOINTER_TYPE = "MemorySaver (fallback)"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def get_checkpointer():
    """
    Returns the module-level checkpointer singleton.

    Currently active: MemorySaver (in-process checkpointing).
    To upgrade to disk persistence: install a langgraph-checkpoint-sqlite
    version compatible with your langgraph version.

    Usage in graph.py:
        from graph.checkpointer import get_checkpointer
        app = workflow.compile(checkpointer=get_checkpointer())

    Returns:
        SqliteSaver instance (if available) or MemorySaver (default).
        Returns None if both fail (graph still runs, just without checkpointing).
    """
    return _checkpointer
