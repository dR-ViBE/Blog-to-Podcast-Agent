# tests/conftest.py
#
# WHY THIS FILE IS THE MOST IMPORTANT TEST FILE:
#   pytest processes conftest.py BEFORE importing any test module.
#   This means we can set environment variables here and they will be in place
#   when test files do `from api.main import app` or `from graph.graph import app`.
#
#   Without this, the following import-time failures would occur in CI:
#     - ChatGroq(model="llama-3.1-8b-instant") → requires GROQ_API_KEY
#     - ElevenLabs(api_key=...) in generate_audio → requires ELEVENLABS_API_KEY
#     - LangSmith startup check → requires LANGCHAIN_API_KEY
#
#   The keys set here are FAKE — they pass format validation but will fail
#   any real API call. Since our tests never make real API calls, this is fine.
#
# EXECUTION ORDER IN PYTEST:
#   1. pytest starts
#   2. conftest.py files are executed (this file first — it's at root of tests/)
#   3. test files are collected (imported)  ← env vars are already set by step 2
#   4. tests are executed

import os

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# FAKE API KEYS — set BEFORE any project module is imported
#
# os.environ.setdefault() only sets the value if the key is NOT already set.
# This means real keys from the shell environment take priority over these fakes.
# In GitHub Actions, these match the `env:` values set in ci.yml.
# ─────────────────────────────────────────────────────────────────────────────

# Groq: ChatGroq validates this key at instantiation time in chain files.
# The "gsk_" prefix matches Groq's key format — avoids format-validation errors.
os.environ.setdefault(
    "GROQ_API_KEY",
    "gsk_testfakekey000000000000000000000000000000000000000000000000000000",
)

# ElevenLabs: checked in generate_audio.py at runtime (not import time).
# Set anyway for completeness and to prevent any future import-time checks.
os.environ.setdefault(
    "ELEVENLABS_API_KEY",
    "sk_testfakekey00000000000000000000000000000000000000000000",
)

# Tavily: used in ingestion.py which is not imported in tests.
# Set for completeness.
os.environ.setdefault(
    "TAVILY_API_KEY",
    "tvly-test-fake-key-00000000000000000000",
)

# LangSmith: LANGCHAIN_TRACING_V2=false disables tracing entirely.
# The API key is still set to prevent any conditional checks from failing.
os.environ.setdefault(
    "LANGCHAIN_API_KEY",
    "lsv2_pt_testfakekey00000000000000000000000000000000",
)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")  # DISABLE tracing in tests
os.environ.setdefault("LANGCHAIN_PROJECT", "podcast-agent-test")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Application metadata (read by services.py for RunnableConfig)
os.environ.setdefault("APP_NAME", "blog-to-podcast-agent")
os.environ.setdefault("APP_VERSION", "1.0.0")
os.environ.setdefault("ENVIRONMENT", "test")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PYTEST FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def fastapi_client():
    """
    Creates a FastAPI TestClient that persists for the entire test session.

    `scope="session"` means the client is created ONCE and shared across all
    tests in all test files. This is more efficient than creating it per-test
    because the FastAPI app startup (lifespan) only runs once.

    WHY TestClient and not a real server?
      TestClient runs the FastAPI app in-process — no network, no port binding.
      It's fast, deterministic, and doesn't require the server to be running.

    WHAT THE LIFESPAN DOES IN TESTS:
      The lifespan in api/main.py creates the outputs/audio/ directory and
      checks LangSmith connectivity. With LANGCHAIN_TRACING_V2=false,
      the LangSmith check is skipped. Directory creation succeeds in any
      writable environment (including the GitHub Actions runner workspace).
    """
    # Import is delayed to here (not at module top) so that env vars above
    # are definitely set before api/main.py and its transitive imports run.
    from fastapi.testclient import TestClient

    from api.main import app

    # Use TestClient as a context manager to trigger the lifespan (startup/shutdown)
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
