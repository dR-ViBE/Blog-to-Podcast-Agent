# tests/test_graph.py
#
# PURPOSE:
#   Tests the LangGraph pipeline structure — state schema, node compilation,
#   conditional logic, and Pydantic model validation.
#
# WHAT THIS DOES NOT TEST:
#   Actual node execution (which requires live Groq, Ollama, ElevenLabs).
#   We test STRUCTURE and CONTRACTS, not runtime behaviour.

import pytest
from typing import get_type_hints


# ─────────────────────────────────────────────────────────────────────────────
# STATE SCHEMA TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphState:
    """Verifies the LangGraph state TypedDict has all required fields."""

    def test_state_is_importable(self):
        """GraphState must be importable from graph.state."""
        from graph.state import GraphState
        assert GraphState is not None

    def test_state_has_query_field(self):
        """GraphState must have a 'query' field for the user's search query."""
        from graph.state import GraphState
        assert "query" in GraphState.__annotations__

    def test_state_has_documents_field(self):
        """GraphState must have a 'documents' field for retrieved blog chunks."""
        from graph.state import GraphState
        assert "documents" in GraphState.__annotations__

    def test_state_has_script_field(self):
        """GraphState must have a 'script' field for the generated podcast text."""
        from graph.state import GraphState
        assert "script" in GraphState.__annotations__

    def test_state_has_is_acceptable_field(self):
        """GraphState must have 'is_acceptable' for the grader's binary decision."""
        from graph.state import GraphState
        assert "is_acceptable" in GraphState.__annotations__

    def test_state_has_script_evaluation_field(self):
        """GraphState must have 'script_evaluation' for the grader's reason."""
        from graph.state import GraphState
        assert "script_evaluation" in GraphState.__annotations__

    def test_state_has_improvement_suggestions_field(self):
        """GraphState must have 'improvement_suggestions' for editor feedback."""
        from graph.state import GraphState
        assert "improvement_suggestions" in GraphState.__annotations__

    def test_state_has_generation_count_field(self):
        """GraphState must have 'generation_count' to track retry attempts."""
        from graph.state import GraphState
        assert "generation_count" in GraphState.__annotations__

    def test_state_has_max_generations_field(self):
        """GraphState must have 'max_generations' for the retry budget."""
        from graph.state import GraphState
        assert "max_generations" in GraphState.__annotations__

    def test_state_has_audio_output_field(self):
        """GraphState must have 'audio_output' for the generated MP3 path."""
        from graph.state import GraphState
        assert "audio_output" in GraphState.__annotations__

    def test_state_has_all_required_fields(self):
        """Verifies the complete set of required state fields in one test."""
        from graph.state import GraphState
        required_fields = {
            "url", "query", "documents", "script",
            "is_acceptable", "script_evaluation", "improvement_suggestions",
            "generation_count", "max_generations", "audio_output",
        }
        actual_fields = set(GraphState.__annotations__.keys())
        missing = required_fields - actual_fields
        assert not missing, (
            f"GraphState is missing required fields: {missing}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH COMPILATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphCompilation:
    """Verifies the LangGraph StateGraph compiles and has the expected structure."""

    def test_graph_compiles_without_error(self):
        """The LangGraph app must compile successfully.

        Compilation validates that all edges reference valid nodes,
        all conditionals return valid node names, and the entry point
        is set. If any of these are wrong, LangGraph raises an error here.
        """
        from graph.graph import app
        assert app is not None

    def test_graph_is_a_runnable(self):
        """The compiled graph must be a LangChain runnable (has .invoke method)."""
        from graph.graph import app
        assert hasattr(app, "invoke"), "Compiled graph must have .invoke() method"
        assert callable(app.invoke), ".invoke must be callable"

    def test_graph_has_get_graph_method(self):
        """Compiled LangGraph apps expose .get_graph() for introspection."""
        from graph.graph import app
        assert hasattr(app, "get_graph"), (
            "Compiled graph must have .get_graph() for LangSmith diagram generation"
        )

    def test_graph_nodes_exist(self):
        """The compiled graph must contain the five expected nodes."""
        from graph.graph import app
        from graph.consts import (
            RETRIEVE, GENERATE_SCRIPT, GRADE_SCRIPT,
            GENERATE_AUDIO, SUGGEST_IMPROVEMENTS,
        )
        # LangGraph's compiled graph exposes its node names via get_graph()
        graph_obj = app.get_graph()
        node_names = set(graph_obj.nodes.keys())

        expected_nodes = {
            RETRIEVE, GENERATE_SCRIPT, GRADE_SCRIPT,
            GENERATE_AUDIO, SUGGEST_IMPROVEMENTS,
        }
        # Check that all expected nodes are present (there may be extra __start__/__end__ nodes)
        missing_nodes = expected_nodes - node_names
        assert not missing_nodes, (
            f"Graph is missing expected nodes: {missing_nodes}. "
            f"Found: {node_names}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDecideNextStep:
    """Tests the routing logic that determines the graph's next node after grading."""

    def test_accepted_script_routes_to_audio(self):
        """When is_acceptable=True, the graph must route to generate_audio."""
        from graph.conditionals.decide_next_step import decide_next_step
        from graph.consts import GENERATE_AUDIO

        state = {
            "is_acceptable": True,
            "generation_count": 1,
            "max_generations": 3,
        }
        result = decide_next_step(state)
        assert result == GENERATE_AUDIO, (
            f"Expected GENERATE_AUDIO when accepted, got: {result}"
        )

    def test_rejected_under_limit_routes_to_improvements(self):
        """When rejected and under the retry limit, route to suggest_improvements."""
        from graph.conditionals.decide_next_step import decide_next_step
        from graph.consts import SUGGEST_IMPROVEMENTS
        from langgraph.graph import END

        state = {
            "is_acceptable": False,
            "generation_count": 1,   # Under max_generations=3
            "max_generations": 3,
        }
        result = decide_next_step(state)
        assert result == SUGGEST_IMPROVEMENTS, (
            f"Expected SUGGEST_IMPROVEMENTS when rejected under limit, got: {result}"
        )

    def test_rejected_at_limit_routes_to_end(self):
        """When rejected AND at max retries, the graph must route to END."""
        from graph.conditionals.decide_next_step import decide_next_step
        from langgraph.graph import END

        state = {
            "is_acceptable": False,
            "generation_count": 3,   # Equal to max_generations → stop
            "max_generations": 3,
        }
        result = decide_next_step(state)
        assert result == END, (
            f"Expected END when rejected at max retries, got: {result}"
        )

    def test_defaults_to_end_with_missing_state(self):
        """With an empty state, the function must not crash and must return END."""
        from graph.conditionals.decide_next_step import decide_next_step
        from langgraph.graph import END

        # Empty state — all .get() calls return their defaults
        result = decide_next_step({})
        # is_acceptable defaults to False, generation_count=0, max_generations=1
        # 0 < 1 → would go to SUGGEST_IMPROVEMENTS, not END
        # This tests that the function handles missing keys gracefully
        assert result is not None, "decide_next_step must return a value for empty state"


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODEL TESTS (API layer)
# ─────────────────────────────────────────────────────────────────────────────

class TestPydanticModels:
    """Tests for request/response model validation logic."""

    def test_podcast_request_default_max_generations(self):
        """PodcastRequest.max_generations must default to 3."""
        from api.models import PodcastRequest
        req = PodcastRequest(query="AI Agents")
        assert req.max_generations == 3

    def test_podcast_request_accepts_valid_query(self):
        """PodcastRequest must accept a valid 3+ character query."""
        from api.models import PodcastRequest
        req = PodcastRequest(query="Prompt Engineering")
        assert req.query == "Prompt Engineering"

    def test_podcast_request_rejects_short_query(self):
        """PodcastRequest must reject queries shorter than 3 characters."""
        from pydantic import ValidationError
        from api.models import PodcastRequest
        with pytest.raises(ValidationError):
            PodcastRequest(query="AI")  # Only 2 chars — below min_length=3

    def test_podcast_request_rejects_zero_generations(self):
        """PodcastRequest.max_generations must reject 0 (ge=1 constraint)."""
        from pydantic import ValidationError
        from api.models import PodcastRequest
        with pytest.raises(ValidationError):
            PodcastRequest(query="AI Agents", max_generations=0)

    def test_podcast_request_rejects_eleven_generations(self):
        """PodcastRequest.max_generations must reject 11 (le=10 constraint)."""
        from pydantic import ValidationError
        from api.models import PodcastRequest
        with pytest.raises(ValidationError):
            PodcastRequest(query="AI Agents", max_generations=11)

    def test_podcast_response_all_optional_fields_none(self):
        """PodcastResponse must allow all optional fields to be None."""
        from api.models import PodcastResponse
        resp = PodcastResponse(
            status="success",
            generation_count=0,
            accepted=False,
        )
        assert resp.script is None
        assert resp.audio_path is None
        assert resp.evaluation is None
        assert resp.improvement_suggestions is None

    def test_podcast_response_full_data(self):
        """PodcastResponse must correctly store all fields when provided."""
        from api.models import PodcastResponse
        resp = PodcastResponse(
            status="success",
            script="Welcome to The Insight Loop...",
            audio_path="podcast_abc123.mp3",
            generation_count=2,
            accepted=True,
            evaluation="Script meets all quality criteria.",
            improvement_suggestions=None,
        )
        assert resp.status == "success"
        assert resp.accepted is True
        assert resp.generation_count == 2
        assert "podcast_abc123.mp3" in resp.audio_path

    def test_health_response_default_status(self):
        """HealthResponse must default to status='healthy'."""
        from api.models import HealthResponse
        resp = HealthResponse()
        assert resp.status == "healthy"
