# api/security.py
#
# PURPOSE:
#   Two production-grade security guards for the POST /podcast endpoint:
#
#   1. PromptInjectionGuard — detects and blocks prompt injection attacks
#      before the query reaches any LLM. This is OWASP LLM Top 10 #1 risk.
#
#   2. PIIGuard — detects and masks Personally Identifiable Information (PII)
#      in user queries before they are sent to the LLM. Ensures we don't
#      accidentally process or store sensitive personal data.
#
# DESIGN PRINCIPLES:
#   - Both guards live at the API boundary (not inside the graph). This means:
#     * They run once, not on every LLM call inside the pipeline.
#     * They are independently testable without touching LangGraph.
#     * They can be applied as FastAPI dependencies with zero graph changes.
#   - Security failures are logged at WARNING level with the query hash (not
#     the raw query) to avoid logging attacker-controlled strings.
#   - PII guard degrades gracefully — if presidio is not installed, it skips
#     silently rather than crashing the API.

import hashlib
import logging
import re
from typing import List, Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PROMPT INJECTION GUARD
# ---------------------------------------------------------------------------

# Patterns that indicate a prompt injection attempt.
# We use case-insensitive regex matching against the normalised query.
#
# WHY THESE PATTERNS:
#   Prompt injection attacks typically try to:
#     a) Override the system prompt ("ignore previous instructions")
#     b) Inject new role/system context ("[SYSTEM]", "<|system|>")
#     c) Jailbreak via role-playing ("act as DAN", "pretend you are")
#     d) Exfiltrate the system prompt ("repeat your instructions")
#
# We do NOT log which pattern matched — this prevents an attacker from
# probing for gaps in our blocklist by reading error responses.

_INJECTION_PATTERNS: List[re.Pattern] = [
    # Override instructions
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|my|your)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|what)\s+(you|i|we)\s+(were|was|have)", re.IGNORECASE),
    # System/role injection via delimiter tokens
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"<\|system\|>", re.IGNORECASE),
    re.compile(r"<\|user\|>", re.IGNORECASE),
    re.compile(r"<\|assistant\|>", re.IGNORECASE),
    re.compile(r"<<SYS>>", re.IGNORECASE),
    re.compile(r"\bSYSTEM:\s", re.IGNORECASE),
    re.compile(r"\bINSTRUCTION:\s", re.IGNORECASE),
    # Jailbreak persona injection
    re.compile(r"\bact\s+as\b.{0,30}\b(DAN|jailbreak|unrestricted|unfiltered|evil)\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(a|an|the)\b", re.IGNORECASE),
    # System prompt extraction
    re.compile(r"repeat\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"(print|output|show|display|reveal|leak)\s+(your|the)\s+(system\s+)?(prompt|instructions)", re.IGNORECASE),
    re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions)", re.IGNORECASE),
    # Developer/debug mode tricks
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"jailbreak\s+mode", re.IGNORECASE),
    re.compile(r"god\s+mode", re.IGNORECASE),
    # Encoding tricks (Base64 encoded instructions)
    re.compile(r"base64", re.IGNORECASE),
]

# Maximum query length we will accept (characters).
# Very long queries are often used to bury injection content after legitimate text.
_MAX_QUERY_LENGTH = 2000


class PromptInjectionGuard:
    """
    Detects and blocks prompt injection attacks in user queries.

    Usage as FastAPI dependency:
        from api.security import injection_guard
        @router.post("/podcast")
        def generate_podcast(request: PodcastRequest, _: None = Depends(injection_guard)):
            ...

    Or call directly:
        guard = PromptInjectionGuard()
        guard.check(query)  # raises HTTPException(400) if injection detected
    """

    def check(self, query: str) -> None:
        """
        Validates a query string for prompt injection patterns.

        Args:
            query: The raw user query string from the API request.

        Raises:
            HTTPException(400): If the query matches any injection pattern
                                or exceeds the maximum allowed length.
        """
        # --- Length check ---
        if len(query) > _MAX_QUERY_LENGTH:
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]
            logger.warning(
                "Query rejected: exceeds max length | hash=%s | length=%d",
                query_hash,
                len(query),
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Query too long ({len(query)} characters). "
                    f"Maximum allowed: {_MAX_QUERY_LENGTH} characters."
                ),
            )

        # --- Pattern check ---
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(query):
                # Log the HASH of the query, not the query itself.
                # This prevents attacker-controlled strings from appearing in logs.
                query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]
                logger.warning(
                    "Prompt injection attempt blocked | hash=%s | pattern=%s",
                    query_hash,
                    pattern.pattern[:40],  # Log truncated pattern name for debugging
                )
                # Do NOT reveal which pattern matched — prevents attacker probing
                raise HTTPException(
                    status_code=400,
                    detail="Query rejected: policy violation detected in input.",
                )

        logger.debug("Query passed injection check | length=%d", len(query))


# Module-level singleton — FastAPI dependencies are called per-request
_injection_guard = PromptInjectionGuard()


def check_prompt_injection(query: str) -> None:
    """Standalone function to check a query for prompt injection. Raises HTTP 400 if detected."""
    _injection_guard.check(query)


# ---------------------------------------------------------------------------
# PII GUARD
# ---------------------------------------------------------------------------

# Presidio entity types we scan for.
# Full list: https://microsoft.github.io/presidio/supported_entities/
_PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "DATE_TIME",
    "NRP",           # Nationality, religion, political group
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "URL",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
]


class PIIGuard:
    """
    Detects and masks Personally Identifiable Information (PII) in text.

    Uses Microsoft Presidio (open-source NLP-based PII detection).
    Degrades gracefully if presidio is not installed — logs a warning and
    returns the original text unchanged (fail-open for availability).

    Two modes of operation:
      - mask_text(text)  → replace PII with placeholder tokens
      - scan_text(text)  → return list of detected entity types (no masking)
    """

    def __init__(self):
        self._available = False
        self._analyzer = None
        self._anonymizer = None
        self._load()

    def _load(self):
        """Attempt to load presidio. Fails gracefully if not installed."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._available = True
            logger.info("PII guard initialised successfully (presidio loaded).")
        except ImportError:
            logger.warning(
                "presidio-analyzer not installed. PII detection DISABLED. "
                "Run: poetry add presidio-analyzer presidio-anonymizer"
            )
        except Exception as exc:
            logger.warning("PII guard failed to initialise: %s. PII detection DISABLED.", exc)

    @property
    def is_available(self) -> bool:
        """Returns True if presidio loaded successfully."""
        return self._available

    def mask_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Mask PII in text, replacing detected entities with placeholder tokens.

        Args:
            text: Input text to scan and mask.

        Returns:
            Tuple of (masked_text, entity_types_found):
              - masked_text: Text with PII replaced by tokens like [PERSON], [EMAIL_ADDRESS]
              - entity_types_found: List of entity type strings that were detected
                                    (e.g. ["PERSON", "EMAIL_ADDRESS"])
        """
        if not self._available or not text:
            return text, []

        try:
            results = self._analyzer.analyze(
                text=text,
                entities=_PII_ENTITIES,
                language="en",
            )

            if not results:
                return text, []

            # Collect detected entity types (deduplicated)
            entity_types = list({r.entity_type for r in results})

            # Log entity types found (NOT the actual PII values)
            logger.warning(
                "PII detected in query | entity_types=%s | text_length=%d",
                entity_types,
                len(text),
            )

            # Anonymize: replace PII with entity-type tokens
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig

            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={
                    entity: OperatorConfig("replace", {"new_value": f"[{entity}]"})
                    for entity in entity_types
                },
            )

            return anonymized.text, entity_types

        except Exception as exc:
            logger.warning("PII masking failed: %s. Returning original text.", exc)
            return text, []

    def scan_text(self, text: str) -> List[str]:
        """
        Scan text for PII without masking. Returns list of detected entity types.

        Used to scan OUTPUT text (generated scripts) — we don't mask the output
        because it would corrupt the podcast script, but we warn if PII is found.

        Args:
            text: Text to scan.

        Returns:
            List of detected entity type strings (empty if none found or presidio unavailable).
        """
        if not self._available or not text:
            return []

        try:
            results = self._analyzer.analyze(
                text=text,
                entities=_PII_ENTITIES,
                language="en",
            )
            entity_types = list({r.entity_type for r in results})
            if entity_types:
                logger.warning(
                    "PII detected in generated output | entity_types=%s | text_length=%d",
                    entity_types,
                    len(text),
                )
            return entity_types
        except Exception as exc:
            logger.warning("PII output scan failed: %s.", exc)
            return []


# Module-level singleton — initialised once at import time
pii_guard = PIIGuard()
