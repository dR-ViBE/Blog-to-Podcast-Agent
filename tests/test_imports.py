# tests/test_imports.py
#
# PURPOSE:
#   Verifies that all major modules and packages can be imported without errors.
#   This catches broken imports, missing dependencies, and misconfigured
#   module paths before any logic is tested.
#
# WHAT THIS DOES NOT TEST:
#   It does not test that API calls work — only that the Python import system
#   can load each module. Real API calls would require valid keys and live services.

import importlib

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY PACKAGE IMPORTS
# Verifies that all packages listed in pyproject.toml are actually installed.
# ─────────────────────────────────────────────────────────────────────────────


def test_import_fastapi():
    """FastAPI must be installed and importable."""
    import fastapi

    assert fastapi.__version__ is not None, "FastAPI version should be set"


def test_import_langgraph():
    """LangGraph must be installed and importable."""
    import langgraph

    assert langgraph is not None


def test_import_langchain():
    """LangChain core must be installed and importable."""
    import langchain

    assert langchain is not None


def test_import_langchain_groq():
    """langchain-groq (Groq LLM integration) must be installed."""
    from langchain_groq import ChatGroq

    assert ChatGroq is not None


def test_import_langchain_ollama():
    """langchain-ollama (embedding model integration) must be installed."""
    from langchain_ollama import OllamaEmbeddings

    assert OllamaEmbeddings is not None


def test_import_langchain_chroma():
    """langchain-chroma (vector store integration) must be installed."""
    from langchain_chroma import Chroma

    assert Chroma is not None


def test_import_langsmith():
    """LangSmith (observability) must be installed."""
    import langsmith

    assert langsmith is not None


def test_import_streamlit():
    """Streamlit (frontend framework) must be installed."""
    import streamlit

    assert streamlit.__version__ is not None


def test_import_uvicorn():
    """Uvicorn (ASGI server) must be installed."""
    import uvicorn

    assert uvicorn is not None


def test_import_elevenlabs():
    """ElevenLabs SDK must be installed."""
    from elevenlabs.client import ElevenLabs

    assert ElevenLabs is not None


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT MODULE IMPORTS
# Verifies that the project's own modules load correctly.
# ─────────────────────────────────────────────────────────────────────────────


def test_import_graph_state():
    """The LangGraph state schema must be importable."""
    from graph.state import GraphState

    assert GraphState is not None


def test_import_graph_consts():
    """Graph node name constants must be importable."""
    from graph.consts import (
        GENERATE_AUDIO,
        GENERATE_SCRIPT,
        GRADE_SCRIPT,
        RETRIEVE,
        SUGGEST_IMPROVEMENTS,
    )

    # Verify the string values haven't been accidentally changed
    assert RETRIEVE == "retrieve"
    assert GENERATE_SCRIPT == "generate_script"
    assert GRADE_SCRIPT == "grade_script"
    assert GENERATE_AUDIO == "generate_audio"
    assert SUGGEST_IMPROVEMENTS == "suggest_improvements"


def test_import_graph_chains():
    """All three LangGraph chains must import without errors.

    This implicitly tests that ChatGroq can be instantiated with the
    fake GROQ_API_KEY set in conftest.py, since the chains instantiate
    ChatGroq at module load time.
    """
    from graph.chains.improvements_chain import suggest_improvements_chain
    from graph.chains.podcast_script_chain import script_generation_chain
    from graph.chains.script_grader_chain import script_grader

    assert script_generation_chain is not None
    assert script_grader is not None
    assert suggest_improvements_chain is not None


def test_import_graph_nodes():
    """All five graph nodes must be importable from the nodes package."""
    from graph.nodes import (
        generate_audio,
        generate_podcast_script,
        grade_script,
        retrieve_blog_chunks,
        suggest_imporvements,  # note: typo is preserved from original code
    )

    assert all(
        [
            generate_audio,
            generate_podcast_script,
            grade_script,
            retrieve_blog_chunks,
            suggest_imporvements,
        ]
    )


def test_import_compiled_graph():
    """The compiled LangGraph StateGraph must be importable.

    This is the most comprehensive import test — it exercises the full
    import chain: graph.py → nodes → chains → ChatGroq instantiation.
    """
    from graph.graph import app as langgraph_app

    assert langgraph_app is not None


def test_import_api_models():
    """FastAPI Pydantic models must be importable."""
    from api.models import HealthResponse, PodcastRequest, PodcastResponse

    assert PodcastRequest is not None
    assert PodcastResponse is not None
    assert HealthResponse is not None


def test_import_api_routes():
    """FastAPI router must be importable."""
    from api.routes import router

    assert router is not None


def test_import_api_services():
    """The service layer must be importable."""
    from api.services import run_podcast_agent

    assert callable(run_podcast_agent)


def test_import_fastapi_app():
    """The FastAPI app instance must be importable.

    This is the full end-to-end import test for the backend:
    api/main.py → api/routes.py → api/services.py → graph/graph.py → (all nodes and chains)
    """
    from api.main import app

    assert app is not None
    assert app.title == "Blog-to-Podcast Agent API"
