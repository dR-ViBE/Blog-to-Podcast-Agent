#!/bin/sh
# docker-entrypoint.sh
#
# This script runs BEFORE the main uvicorn process starts.
# Its job is to ensure the Ollama embedding model is downloaded and ready,
# then hand off execution to whatever CMD is specified (uvicorn).
#
# WHY THIS EXISTS:
#   When `docker compose up` starts all services simultaneously, the `api`
#   container becomes healthy before the Ollama model is pulled. Without
#   this script the first request that uses OllamaEmbeddings would fail
#   because the model doesn't exist yet.
#
#   The `depends_on: ollama-pull: condition: service_completed_successfully`
#   in docker-compose.yml already waits for the model pull to finish, but
#   this script adds a secondary safety check and informative logs.
#
# USAGE:
#   This file is copied into the container and set as ENTRYPOINT in Dockerfile.api.
#   It receives the CMD arguments via "$@" and exec's them at the end.

set -e  # Exit immediately if any command fails

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"

echo "============================================================"
echo "  Blog-to-Podcast Agent — FastAPI Startup"
echo "============================================================"
echo "  OLLAMA_HOST : ${OLLAMA_HOST}"
echo "  APP_VERSION : ${APP_VERSION:-unknown}"
echo "  ENVIRONMENT : ${ENVIRONMENT:-development}"
echo "============================================================"

# ── Wait for Ollama server to be reachable ───────────────────────────────────
# Poll the Ollama /api/tags endpoint (lists available models) until it responds.
# Timeout after 120 seconds to avoid hanging indefinitely.
echo "[entrypoint] Waiting for Ollama server at ${OLLAMA_HOST}..."
MAX_WAIT=120
WAITED=0
until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "[entrypoint] ERROR: Ollama did not become ready within ${MAX_WAIT}s."
        echo "[entrypoint] Check that the 'ollama' Docker service is running."
        exit 1
    fi
    echo "[entrypoint] Ollama not ready yet — retrying in 3s... (${WAITED}s elapsed)"
    sleep 3
    WAITED=$((WAITED + 3))
done
echo "[entrypoint] Ollama is ready."

# ── Verify the embedding model is available ───────────────────────────────────
# The nomic-embed-text model is pulled by the `ollama-pull` service in
# docker-compose.yml. We just verify it's there before starting FastAPI.
echo "[entrypoint] Checking for nomic-embed-text model..."
if curl -sf "${OLLAMA_HOST}/api/tags" | grep -q "nomic-embed-text"; then
    echo "[entrypoint] nomic-embed-text model is available. Starting API..."
else
    echo "[entrypoint] WARNING: nomic-embed-text model not found."
    echo "[entrypoint] The ollama-pull service may still be downloading it."
    echo "[entrypoint] Proceeding anyway — embeddings will fail until the model is ready."
fi

echo "============================================================"
echo "  Starting: $@"
echo "============================================================"

# Hand off to the CMD (uvicorn api.main:app ...)
# `exec` replaces this shell with the uvicorn process so signals (SIGTERM, etc.)
# are forwarded correctly — important for graceful Docker shutdown.
exec "$@"
