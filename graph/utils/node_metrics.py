# graph/utils/node_metrics.py
#
# PURPOSE:
#   Provides a lightweight context manager for timing individual LangGraph
#   nodes and recording those timings into Prometheus metrics.
#
# PHASE 2 (#10) — PER-NODE METRICS:
#   Each graph node wraps its main chain.invoke() call with NodeTimer:
#
#       from graph.utils.node_metrics import NodeTimer
#
#       with NodeTimer("generate_script"):
#           result = chain.invoke(inputs, config=node_config)
#
#   On exit, NodeTimer records:
#     - podcast_node_duration_seconds{node_name="generate_script"}
#
#   For nodes that return cost data (via state), services.py records
#   podcast_node_llm_cost_usd_total separately (per-node cost is
#   harder to attribute at node level without callback hooks).
#
# WHY A CONTEXT MANAGER?
#   Context managers guarantee the observe() call runs even if the chain
#   raises an exception — we still want to record the time spent in a
#   failing node for debugging. The `success` label distinguishes good
#   runs from erroring ones in Prometheus queries.
#
# THREAD SAFETY:
#   time.monotonic() is thread-safe. METRICS is a module-level singleton
#   with thread-safe Prometheus label access.

import logging
import time
from types import TracebackType
from typing import Optional, Type

logger = logging.getLogger(__name__)


class NodeTimer:
    """
    Context manager that times a LangGraph node and records the duration
    into the `podcast_node_duration_seconds` Prometheus histogram.

    Usage:
        with NodeTimer("plan_episode"):
            result = planner_chain.invoke(inputs)

    The timer starts on __enter__ and records elapsed time on __exit__,
    regardless of whether the node succeeded or raised an exception.

    Args:
        node_name: A short identifier for the graph node being timed.
                   Should match the node name registered in graph.py
                   (e.g. "plan_episode", "retrieve", "generate_script",
                    "grade_script", "suggest_improvements", "generate_audio").
    """

    def __init__(self, node_name: str) -> None:
        self.node_name = node_name
        self._start: float = 0.0

    def __enter__(self) -> "NodeTimer":
        self._start = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Records elapsed time to Prometheus on context exit (success or failure)."""
        elapsed = time.monotonic() - self._start

        try:
            # Import here to avoid circular imports at module load time
            # (metrics.py imports nothing from graph/, so this is safe).
            from api.metrics import METRICS

            METRICS.node_duration_seconds.labels(node_name=self.node_name).observe(elapsed)

            logger.debug(
                "Node timed | node=%s | elapsed=%.3fs | success=%s",
                self.node_name,
                elapsed,
                exc_type is None,
            )
        except Exception as metrics_exc:
            # Never let metrics recording crash the pipeline
            logger.warning(
                "Failed to record node duration metric | node=%s | error=%s",
                self.node_name,
                metrics_exc,
            )
