# api/main.py
#
# This is the FastAPI application entry point.
#
# WHY A SEPARATE main.py FOR THE APP?
#   Keeping the FastAPI `app` instance in its own file means:
#     - Routes, models, and services can be imported without side effects.
#     - The app can be imported cleanly by test frameworks (pytest, etc.)
#       without accidentally starting the server.
#     - Uvicorn's reload mechanism targets exactly this file:
#       `uvicorn api.main:app --reload`

import logging
import logging.config
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables before any internal modules are imported
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# langsmith is imported here only for the startup health check.
# It does NOT need to be called explicitly anywhere else —
# setting LANGCHAIN_TRACING_V2=true in .env is all that is required
# for automatic tracing of LangChain and LangGraph calls.
try:
    from langsmith import Client as LangSmithClient

    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False

from api.routes import router


# ---------------------------------------------------------------------------
# LOGGING CONFIGURATION
#
# We configure logging once here at the application entry point.
# Using Python's `logging` module (not print()) means:
#   - Log levels can be controlled per environment (DEBUG in dev, INFO in prod).
#   - Structured loggers in each module (logger = logging.getLogger(__name__))
#     automatically inherit this configuration.
#   - Logs can later be forwarded to CloudWatch, Datadog, etc. by swapping
#     handlers without touching any module code.
# ---------------------------------------------------------------------------
LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,  # Don't silence third-party library logs
    "formatters": {
        "default": {
            # ISO-style timestamp | level | module:line | message
            "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
    # Quiet noisy third-party libraries in the log output
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "chromadb": {"level": "WARNING"},
        "uvicorn.access": {"level": "INFO"},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

# Module logger — used for startup/shutdown messages from this file
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LIFESPAN (startup / shutdown events)
#
# FastAPI's `lifespan` context manager replaces the deprecated
# `@app.on_event("startup")` pattern.  Code before `yield` runs on startup,
# code after `yield` runs on shutdown.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
        - Ensures the `outputs/audio/` directory exists so the generate_audio
          node never fails due to a missing directory.
        - Logs a ready message so operators know the server is initialised.

    Shutdown:
        - Logs a shutdown message (add cleanup logic here if needed, e.g.
          closing a database connection pool).
    """
    # --- STARTUP ---
    audio_dir = Path("outputs/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Audio output directory ready | path=%s", audio_dir.resolve())

    # ── LangSmith startup verification ──────────────────────────────────────
    # We check at startup whether tracing is configured correctly so that
    # any misconfiguration is caught immediately (not silently on first request).
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_key_set = bool(os.getenv("LANGCHAIN_API_KEY"))
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "(not set)")

    if tracing_enabled and langsmith_key_set:
        logger.info(
            "LangSmith tracing ENABLED | project=%s | endpoint=%s",
            langsmith_project,
            os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        )
        # Optionally verify the API key by listing projects (catches bad keys early)
        if _LANGSMITH_AVAILABLE:
            try:
                client = LangSmithClient()
                # list_projects() makes a lightweight API call to verify credentials
                _ = list(client.list_projects())
                logger.info("LangSmith credentials verified successfully.")
            except Exception as exc:
                # Don't crash the server — just warn the operator
                logger.warning("LangSmith credential check failed (tracing may not work): %s", exc)
    elif tracing_enabled and not langsmith_key_set:
        logger.warning(
            "LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set. "
            "Tracing will be silently disabled by LangChain."
        )
    else:
        logger.info("LangSmith tracing DISABLED. Set LANGCHAIN_TRACING_V2=true in .env to enable.")
    # ── End LangSmith startup check ─────────────────────────────────────────

    logger.info("Blog-to-Podcast API is ready to accept requests.")

    yield  # Application runs here

    # --- SHUTDOWN ---
    logger.info("Blog-to-Podcast API is shutting down.")


# ---------------------------------------------------------------------------
# FASTAPI APP INSTANCE
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Blog-to-Podcast Agent API",
    description=(
        "A production-ready REST API that wraps a LangGraph pipeline. "
        "Given a query, it retrieves blog content from a ChromaDB vector store, "
        "generates a podcast script with Groq (LLaMA 3.1), iteratively improves "
        "it using a self-grading loop, and synthesises speech via ElevenLabs."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Docs are served at /docs (Swagger UI) and /redoc (ReDoc) by default.
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ---------------------------------------------------------------------------
# CORS MIDDLEWARE
#
# Cross-Origin Resource Sharing (CORS) headers allow a browser-based frontend
# (e.g., a React app on localhost:3000) to call this API (on localhost:8000).
# In production, replace allow_origins=["*"] with your specific frontend URL.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REGISTER ROUTES
#
# All API routes live in routes.py and are attached here via include_router().
# The prefix="" means routes are served at the root: /health, /podcast, etc.
# ---------------------------------------------------------------------------
app.include_router(router, prefix="")


# ---------------------------------------------------------------------------
# PROMETHEUS METRICS ENDPOINT
#
# /metrics is the standard Prometheus scrape endpoint. Prometheus (or any
# compatible scraper) polls this URL periodically to collect all metric values.
#
# HOW IT WORKS:
#   - prometheus_fastapi_instrumentator auto-instruments all FastAPI routes
#     (http_request_duration_seconds, http_requests_total, etc.)
#   - Our custom metrics (api/metrics.py) are also exposed here automatically
#     because they are registered in the same prometheus_client registry.
#
# USAGE:
#   curl http://localhost:8000/metrics
#   → Returns all metrics in Prometheus text exposition format
# ---------------------------------------------------------------------------
Instrumentator(
    should_group_status_codes=True,  # Group 2xx/4xx/5xx instead of per-code
    should_ignore_untemplated=True,   # Ignore unknown routes (reduces cardinality)
    should_respect_env_var=True,      # Disable via ENABLE_METRICS=false if needed
    env_var_name="ENABLE_METRICS",
    excluded_handlers=["/metrics", "/health"],  # Don't instrument the metrics endpoint itself
).instrument(app).expose(app, endpoint="/metrics", tags=["Observability"])


# ---------------------------------------------------------------------------
# ROOT REDIRECT (optional convenience)
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """Redirect API root to the interactive Swagger UI docs."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")
