# 📊 Prometheus Metrics Dashboard Guide
## Blog-to-Podcast Agent — Observability Reference

**Endpoint:** `GET http://localhost:8000/metrics`  
**Format:** Prometheus text exposition format (scrape-ready)

---

## How Prometheus Metrics Work

Prometheus uses a **pull model**: your Prometheus server scrapes `GET /metrics` every N seconds (default: 15s). Each metric value is a timestamped data point. Grafana visualises these as time-series graphs.

**Metric Types:**
- **Counter** — always increases. Use `rate(metric[5m])` to get events/sec over 5 minutes.
- **Histogram** — tracks distribution across buckets. Use `histogram_quantile(0.95, ...)` for p95 latency.
- **Gauge** — current value (can go up and down). Read directly.

---

## 🟢 Request & Pipeline Metrics

### `podcast_requests_total` (Counter)
**Labels:** `status` = `success` | `failed` | `no_audio`

**What it measures:** Total POST /podcast requests, categorised by outcome.

| Status | Meaning |
|---|---|
| `success` | Script accepted, audio generated |
| `no_audio` | Script accepted but audio generation failed |
| `failed` | Pipeline error (LLM, graph, or unexpected exception) |

**Prometheus query:** `rate(podcast_requests_total[5m])` — requests per second  
**Grafana panel type:** Time series (stacked by status label)

| Value | What it means |
|---|---|
| All `success` | ✅ System healthy |
| High `failed` rate (> 10%) | ❌ Check LLM API keys or graph logic |
| Sustained `no_audio` | ⚠️ ElevenLabs quota exceeded, check TTS fallback |

---

### `podcast_pipeline_duration_seconds` (Histogram)
**Labels:** none (dimensional slicing via quantile)

**What it measures:** End-to-end time from request received to response sent. Includes all LLM calls, retrieval, reranking, and audio generation.

**Prometheus query:**
```promql
histogram_quantile(0.50, rate(podcast_pipeline_duration_seconds_bucket[5m]))  # p50 (median)
histogram_quantile(0.95, rate(podcast_pipeline_duration_seconds_bucket[5m]))  # p95
histogram_quantile(0.99, rate(podcast_pipeline_duration_seconds_bucket[5m]))  # p99
```

**Grafana panel type:** Time series with p50/p95/p99 lines

| Value | What it means |
|---|---|
| p50 = 20-40s | ✅ Normal — LLM + TTS is inherently slow |
| p95 < 90s | ✅ Good |
| p95 > 120s | ⚠️ LLM timeout or overload, check Groq API status |
| p99 > 180s | ❌ Critical — requests likely timing out on client side |

**Why this matters:** This is the most important UX metric. High latency = frustrated users.

---

### `podcast_generation_attempts_total` (Counter)
**Labels:** `result` = `accepted` | `rejected_revise` | `rejected_context` | `max_reached`

**What it measures:** Script generation loop iterations across all runs.

**Prometheus query:** `rate(podcast_generation_attempts_total[10m])`

| Result label | Meaning |
|---|---|
| `accepted` | Script passed grader on this attempt |
| `rejected_revise` | Editor said "fix the writing" (Writer retry) |
| `rejected_context` | Editor said "get more content" (Retriever retry) |
| `max_reached` | Hit max_generations without acceptance |

| Pattern | What it means |
|---|---|
| Mostly `accepted` | ✅ Pipeline quality is high |
| High `rejected_revise` (> 30%) | ⚠️ Writer or grader prompts need tuning |
| High `rejected_context` (> 20%) | ⚠️ ChromaDB corpus is thin — ingest more content |
| High `max_reached` (> 15%) | ❌ Pipeline consistently failing to produce acceptable scripts |

---

## 💰 Cost & Token Metrics

### `podcast_llm_cost_usd_total` (Counter)
**Labels:** `model` = `llama-3.1-8b-instant`

**What it measures:** Cumulative estimated LLM cost in USD since last restart.

**Prometheus query:**
```promql
rate(podcast_llm_cost_usd_total[1h]) * 3600  # USD per hour
rate(podcast_llm_cost_usd_total[1h]) * 3600 * 24  # USD per day (projected)
```

> ⚠️ **Accuracy Note:** These are text-length estimates (~4 chars/token approximation). Accuracy is ~92-95% vs exact token counts. Do not use for billing — treat as a monitoring signal.

| Value | What it means |
|---|---|
| $0.0001-$0.002 per run | ✅ Normal for llama-3.1-8b-instant on Groq |
| $0.003+ per run | ⚠️ Query has very large context (long blog posts) |
| Sudden spike | ❌ Runaway generation loop or unusually long inputs |

**Grafana panel type:** Stat (current rate) + Time series (cost over time)

---

### `podcast_tokens_used_total` (Counter)
**Labels:** `model`, `token_type` = `prompt` | `completion`

**What it measures:** Total tokens consumed, split by prompt vs completion.

**Prometheus query:**
```promql
rate(podcast_tokens_used_total{token_type="prompt"}[5m])
rate(podcast_tokens_used_total{token_type="completion"}[5m])
```

**High prompt/completion ratio (> 3:1):** Prompt bloat — consider shortening context passed to LLMs.

---

## ✍️ Script Quality Metrics

### `podcast_scripts_accepted_total` (Counter)
**Labels:** `first_attempt` = `True` | `False`

**What it measures:** How many scripts pass the Editor on the first attempt vs after revisions.

**Prometheus query:**
```promql
rate(podcast_scripts_accepted_total{first_attempt="True"}[1h]) /
rate(podcast_scripts_accepted_total[1h])  # First-attempt acceptance rate
```

| Value | What it means |
|---|---|
| First-attempt rate > 70% | ✅ Writer and Planner are working well together |
| First-attempt rate 40-70% | ⚠️ Multiple revisions needed — tune prompts |
| First-attempt rate < 40% | ❌ Writer prompt or grader criteria need overhaul |

---

### `podcast_script_word_count` (Histogram)

**What it measures:** Distribution of word counts for accepted scripts.

**Prometheus query:**
```promql
histogram_quantile(0.50, podcast_script_word_count_bucket)  # Median script length
```

| Bucket range | Target | Meaning |
|---|---|---|
| < 350 words | ❌ Too short | Scripts < 3 minutes — listener experience poor |
| 450-750 words | ✅ Ideal | 3-5 minute podcast episode |
| > 900 words | ⚠️ Too long | May exceed TTS API limits or listener attention |

---

## 🛡️ Security Metrics

### `podcast_injection_attempts_total` (Counter)
**Labels:** `reason` = `pattern_match` | `length_exceeded`

**What it measures:** Prompt injection attacks blocked by the security guard.

**Prometheus query:** `rate(podcast_injection_attempts_total[5m])`

| Value | What it means |
|---|---|
| ~0 | ✅ Normal — no active attack |
| Occasional spikes | ⚠️ Bot scanning or curious users probing the API |
| Sustained high rate | ❌ Active attack — consider IP rate limiting or blocking |

> 🔒 **Note:** Individual attack patterns are logged at WARNING level with query hash only (not the raw query) to avoid logging attacker-controlled strings.

---

### `podcast_pii_detections_total` (Counter)
**Labels:** `entity_type` = `PERSON` | `EMAIL_ADDRESS` | `PHONE_NUMBER` | `CREDIT_CARD` | etc.

**What it measures:** PII entities detected and masked in user input queries.

| Entity type | Expected frequency | Alert if |
|---|---|---|
| `PERSON` | Moderate — research queries often mention names | > 50/hour |
| `EMAIL_ADDRESS` | Low | Any sustained count |
| `PHONE_NUMBER` | Low | Any count |
| `CREDIT_CARD` | Should be zero | Any count → ❌ investigate immediately |
| `US_SSN` | Should be zero | Any count → ❌ investigate immediately |

---

### `podcast_pii_output_detections_total` (Counter)
**Labels:** `entity_type`

**What it measures:** PII found in *generated* podcast scripts (output scanning).

| Value | What it means |
|---|---|
| Always 0 | ✅ No PII leaking from ingested documents into outputs |
| Any count | ❌ Review ingested content — source documents contain PII |

> This metric catches ingested PDFs or blog posts that contain personal data (e.g. meeting notes with names, emails in footers).

---

## 🔍 Retrieval Metrics

### `podcast_retrieval_tool_calls_total` (Counter)
**Labels:** `tool` = `search_vectorstore` | `search_web`

**What it measures:** How often each retrieval tool is called by the Retriever Agent.

**Prometheus query:** `rate(podcast_retrieval_tool_calls_total[10m])`

| Pattern | What it means |
|---|---|
| Mostly `search_vectorstore` | ✅ Local corpus is rich and covering queries |
| High `search_web` (> 30%) | ⚠️ ChromaDB corpus too thin — ingest more content |
| Only `search_web` | ❌ Vector store empty or retrieval broken |

---

### `podcast_retrieval_docs_found` (Histogram)

**What it measures:** Unique documents found per retrieval run (after dedup, before reranking).

**Prometheus query:**
```promql
histogram_quantile(0.50, rate(podcast_retrieval_docs_found_bucket[5m]))  # Median docs found
```

| Value | What it means |
|---|---|
| Median < 2 | ❌ Corpus too sparse — ingest more content |
| Median 3-8 | ✅ Good retrieval coverage |
| Median > 12 | ⚠️ Too many results — check deduplication |

---

## 📥 Ingestion Metrics

### `podcast_ingestion_chunks_total` (Counter)
**Labels:** `source_type` = `url` | `pdf` | `text` | `directory`

**What it measures:** Document chunks added to ChromaDB, by content type.

**Prometheus query:** `podcast_ingestion_chunks_total`

**Grafana panel type:** Pie chart (corpus composition by source type)

This metric shows the composition of your knowledge base over time.

---

### `podcast_ingestion_requests_total` (Counter)
**Labels:** `source_type`, `status` = `success` | `failed`

**What it measures:** Total ingestion API calls and their outcomes.

| Pattern | What it means |
|---|---|
| All `success` | ✅ Ingestion pipeline healthy |
| High `failed` rate | ❌ Check file format support or URL accessibility |

---

## 🖥️ System Health

### `podcast_active_pipeline_runs` (Gauge)

**What it measures:** Currently executing pipeline runs (in-flight requests).

| Value | What it means |
|---|---|
| 0 | ✅ Idle |
| 1-3 | ✅ Normal under load |
| > 5 sustained | ⚠️ Requests backing up — consider async endpoint |
| Stuck > 0 for hours | ❌ Pipeline hung — restart API service |

---

## HTTP Auto-Metrics (from prometheus-fastapi-instrumentator)

These are automatically collected for **all endpoints**:

| Metric | Type | Description |
|---|---|---|
| `http_requests_total` | Counter | Requests by method, endpoint, status code group |
| `http_request_duration_seconds` | Histogram | Latency by endpoint |

**Example queries:**
```promql
# Error rate across all endpoints
rate(http_requests_total{status="5xx"}[5m]) / rate(http_requests_total[5m])

# /podcast endpoint latency p95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="/podcast"}[5m]))
```

---

## Suggested Grafana Dashboard Layout

```
Row 1: System Overview
├── [Stat] Active runs
├── [Stat] Requests/min
├── [Stat] Success rate %
└── [Stat] Avg cost per run

Row 2: Latency
├── [Time series] Pipeline duration p50/p95/p99
└── [Histogram] Duration distribution

Row 3: Quality
├── [Time series] First-attempt acceptance rate
├── [Histogram] Script word count distribution
└── [Time series] Generation attempts by result

Row 4: Cost & Tokens
├── [Time series] Cumulative cost (USD/hour)
├── [Time series] Token usage by type
└── [Stat] Total cost today

Row 5: Security
├── [Time series] Injection attempts
├── [Bar] PII detections by entity type
└── [Stat] PII output detections (should be 0)

Row 6: Retrieval
├── [Pie] Tool calls (vectorstore vs web)
├── [Histogram] Docs found per retrieval
└── [Pie] Ingested corpus by source_type
```
