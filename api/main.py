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
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

# ---------------------------------------------------------------------------
# Load .env before anything else
# This ensures GROQ_API_KEY, ELEVENLABS_API_KEY, TAVILY_API_KEY, etc. are
# available as environment variables when the graph modules are imported.
# ---------------------------------------------------------------------------
load_dotenv()


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
    allow_origins=["*"],          # TODO: restrict to your frontend domain in prod
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
# ROOT REDIRECT (optional convenience)
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """Redirect API root to the interactive Swagger UI docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
