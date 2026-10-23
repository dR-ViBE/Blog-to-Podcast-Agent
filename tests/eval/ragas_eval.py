"""
tests/eval/ragas_eval.py

PURPOSE:
    Standalone evaluation script for the Blog-to-Podcast RAG pipeline.
    Uses RAGAS (Retrieval-Augmented Generation Assessment) to measure
    both retrieval quality and generation quality.

METRICS EVALUATED (Phase 1 — expanded from 2 to 6):

  RETRIEVAL QUALITY (measures how good the retrieved context is):
    - context_precision     : Of retrieved chunks, what % are actually relevant?
                              Range: 0.0-1.0. Target: > 0.75. Bad if < 0.50.
    - context_recall        : Of relevant information, what % was retrieved?
                              Range: 0.0-1.0. Target: > 0.70. Bad if < 0.50.
    - context_entity_recall : Are key named entities from ground truth present in retrieved chunks?
                              Range: 0.0-1.0. Target: > 0.60. Bad if < 0.40.

  GENERATION QUALITY (measures how good the LLM's answer is — requires running the full graph):
    - answer_relevancy      : Does the generated script actually address the query?
                              Range: 0.0-1.0. Target: > 0.80. Bad if < 0.65.
    - faithfulness          : Does the script stick to retrieved context (no hallucination)?
                              Range: 0.0-1.0. Target: > 0.75. Bad if < 0.60.
                              Low score = model is making things up beyond the context.
    - noise_sensitivity     : How much does irrelevant context hurt generation quality?
                              Range: 0.0-1.0. Lower is BETTER (less sensitive to noise = more robust).
                              Target: < 0.30. Bad if > 0.50.

WHAT EACH METRIC MEANS FOR AN INTERVIEWER:
    - context_precision + recall  → "We validate our retrieval pipeline quantitatively"
    - faithfulness                → "We actively measure and minimise hallucination"
    - answer_relevancy            → "We verify the output actually answers the user's question"
    - context_entity_recall       → "We check that key facts make it from corpus to context"
    - noise_sensitivity           → "We test system robustness to irrelevant retrieved content"

USAGE:
    # Run full evaluation (requires .chroma populated and Ollama running)
    poetry run python -m tests.eval.ragas_eval

    # Run retrieval-only evaluation (faster — no graph invocation)
    poetry run python -m tests.eval.ragas_eval --retrieval-only

REQUIREMENTS:
    - ChromaDB must be populated via ingestion (run ingestion first)
    - GROQ_API_KEY must be set in .env (used as the judge LLM)
    - Ollama must be running locally (for embedding queries)
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder

from graph.chains.retriever_chain import retriever_chain

# ─── Configuration ────────────────────────────────────────────────────────────

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_OUTPUT_PATH = Path(__file__).parent / "ragas_results.json"

CHROMA_COLLECTION = "blog_podcast_agent"
CHROMA_DIR = "./.chroma"
EMBEDDING_MODEL = "nomic-embed-text"
RERANKER_MODEL = "BAAI/bge-reranker-base"
TOP_K_PER_QUERY = 4
FINAL_TOP_K = 6

# ─── Metric documentation ────────────────────────────────────────────────────
# Each entry: (metric_name, target, bad_threshold, higher_is_better, interpretation)
METRIC_GUIDE = {
    "context_precision": {
        "target": 0.75,
        "bad_threshold": 0.50,
        "higher_is_better": True,
        "description": "Of retrieved chunks, what % are actually relevant to the query?",
        "interpretation": "Low = retriever is fetching too many irrelevant chunks (noise).",
    },
    "context_recall": {
        "target": 0.70,
        "bad_threshold": 0.50,
        "higher_is_better": True,
        "description": "Of all relevant information, what % was retrieved?",
        "interpretation": "Low = retriever is missing relevant content (coverage gap).",
    },
    "context_entity_recall": {
        "target": 0.60,
        "bad_threshold": 0.40,
        "higher_is_better": True,
        "description": "Are key named entities from ground truth present in retrieved context?",
        "interpretation": "Low = specific facts/entities are missing from retrieved chunks.",
    },
    "answer_relevancy": {
        "target": 0.80,
        "bad_threshold": 0.65,
        "higher_is_better": True,
        "description": "Does the generated script actually address the user's query?",
        "interpretation": "Low = LLM is drifting off-topic or not answering the question.",
    },
    "faithfulness": {
        "target": 0.75,
        "bad_threshold": 0.60,
        "higher_is_better": True,
        "description": "Does the generated script stick to retrieved context (no hallucination)?",
        "interpretation": "Low = LLM is fabricating claims not supported by the retrieved context.",
    },
    "noise_sensitivity": {
        "target": 0.30,
        "bad_threshold": 0.50,
        "higher_is_better": False,
        "description": "How much does irrelevant retrieved context hurt generation quality?",
        "interpretation": "High = system is fragile to noisy retrieval. Lower is better.",
    },
}


def run_retrieval_pipeline(query: str) -> List[str]:
    """
    Runs the exact same retrieval pipeline as the production node:
      1. retriever_chain  → generates multi-query strategy from query
      2. ChromaDB         → similarity search for each sub-query
      3. Deduplication    → removes duplicate chunks
      4. Reranking        → BGE cross-encoder scores and ranks chunks

    Returns:
        List of retrieved page_content strings (the contexts).
    """
    minimal_outline = {"episode_title": query, "key_talking_points": [{"topic": query}]}
    strategy = retriever_chain.invoke({"outline": json.dumps(minimal_outline, indent=2)})

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
    )

    all_docs = []
    search_queries = strategy.queries if strategy.queries else [query]
    for q in search_queries:
        docs = vectorstore.similarity_search(query=q, k=TOP_K_PER_QUERY)
        all_docs.extend(docs)

    # Deduplication
    seen = set()
    unique_docs = []
    for doc in all_docs:
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            unique_docs.append(doc)

    # Reranking
    if not unique_docs:
        return []

    model = CrossEncoder(RERANKER_MODEL)
    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = model.predict(pairs)
    scored = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    final_docs = [doc for doc, _ in scored[:FINAL_TOP_K]]

    return [doc.page_content for doc in final_docs]


def run_full_pipeline(query: str) -> tuple:
    """
    Runs the complete Blog-to-Podcast pipeline for faithfulness/answer_relevancy evaluation.
    Returns (retrieved_contexts, generated_script).
    """
    contexts = run_retrieval_pipeline(query)

    try:
        # Import graph only when needed (requires API keys)
        from graph.graph import app as langgraph_app

        initial_state = {
            "query": query,
            "max_generations": 1,   # Single generation for eval efficiency
            "generation_count": 0,
            "source_filter": None,
            "source_type_filter": None,
            "llm_cost_usd": 0.0,
            "total_tokens_used": 0,
            "pii_was_masked": False,
            "pii_entities_found": [],
        }
        final_state = langgraph_app.invoke(initial_state)
        script = final_state.get("script", "")
        return contexts, script
    except Exception as e:
        print(f"    [WARNING] Full pipeline run failed: {e}")
        return contexts, ""


def _print_metric_summary(metric: str, score: float):
    """Print a metric with its value, bar, and status (✓ / ⚠ / ✗)."""
    guide = METRIC_GUIDE.get(metric, {})
    target = guide.get("target", 0.75)
    bad_threshold = guide.get("bad_threshold", 0.50)
    higher_is_better = guide.get("higher_is_better", True)

    if math.isnan(score):
        status = "⚠ NaN  [Evaluation timed out/failed for this metric]"
        bar = "?" * 20
    elif higher_is_better:
        if score >= target:
            status = "✓ GOOD"
        elif score >= bad_threshold:
            status = "⚠ FAIR"
        else:
            status = "✗ BAD"
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
    else:
        # Lower is better (noise_sensitivity)
        if score <= target:
            status = "✓ GOOD"
        elif score <= bad_threshold:
            status = "⚠ FAIR"
        else:
            status = "✗ BAD"
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))

    label = metric.replace("_", " ").title()
    print(f"  {label:<28} {score:.4f}  [{bar}]  {status}")


def evaluate(retrieval_only: bool = False):
    """
    Main evaluation loop.
    Loads the golden dataset, runs retrieval (and optionally full pipeline)
    for each query, then computes all RAGAS metrics.
    """
    print("=" * 65)
    print("  RAGAS Evaluation — Blog-to-Podcast RAG Pipeline")
    print(f"  Mode: {'Retrieval Only' if retrieval_only else 'Full Pipeline (includes generation metrics)'}")
    print("=" * 65)

    with open(GOLDEN_DATASET_PATH) as f:
        golden = json.load(f)

    dataset_entries = golden["dataset"]
    print(f"\nLoaded {len(dataset_entries)} queries from golden dataset.\n")

    # ── Collect samples ────────────────────────────────────────────────────────
    samples = []
    for i, entry in enumerate(dataset_entries, 1):
        query = entry["query"]
        ground_truths = entry["ground_truths"]
        query_type = entry.get("query_type", "unknown")

        print(f"[{i}/{len(dataset_entries)}] ({query_type}) '{query[:60]}...'")

        if retrieval_only:
            try:
                contexts = run_retrieval_pipeline(query)
                print(f"    → Retrieved {len(contexts)} chunks")
                script = ""
            except Exception as e:
                print(f"    [WARNING] Retrieval failed: {e}")
                contexts, script = [], ""
        else:
            try:
                contexts, script = run_full_pipeline(query)
                print(f"    → Retrieved {len(contexts)} chunks | script={len(script.split())} words")
            except Exception as e:
                print(f"    [WARNING] Pipeline failed: {e}")
                contexts, script = [], ""

        samples.append({
            "user_input": query,
            "retrieved_contexts": contexts,
            "reference": " ".join(ground_truths),
            "response": script or "(generation skipped)",
            "query_type": query_type,
        })

    # ── Run RAGAS scoring ─────────────────────────────────────────────────────
    print("\nRunning RAGAS scoring (LLM-as-judge via Groq)...")
    print("This may take 2-5 minutes.\n")

    try:
        from langchain_groq import ChatGroq
        from langchain_ollama import OllamaEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import ContextPrecision, ContextRecall, ContextEntityRecall
        from ragas.run_config import RunConfig

        judge_llm = ChatGroq(model="llama-3.1-8b-instant")
        judge_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # Build RAGAS dataset
        ragas_samples = [
            SingleTurnSample(
                user_input=s["user_input"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s["reference"],
                response=s["response"],
            )
            for s in samples
        ]
        eval_dataset = EvaluationDataset(samples=ragas_samples)

        # ── Select metrics based on mode ──────────────────────────────────────
        # Retrieval metrics (always run)
        metrics = [ContextPrecision(), ContextRecall(), ContextEntityRecall()]

        # Generation metrics (only when we have actual responses)
        if not retrieval_only:
            try:
                from ragas.metrics import AnswerRelevancy, Faithfulness, NoiseSensitivity
                metrics.extend([AnswerRelevancy(), Faithfulness(), NoiseSensitivity()])
                print("Running all 6 metrics (retrieval + generation).")
            except ImportError as e:
                print(f"[WARNING] Generation metrics not available in this RAGAS version: {e}")
                print("Running retrieval-only metrics (3 metrics).")
        else:
            print("Running 3 retrieval metrics only.")

        run_config = RunConfig(max_workers=1, timeout=300)

        result = ragas_evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
        )

        # ── Display Results ────────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print("  RAGAS RESULTS — Phase 1 Evaluation")
        print("=" * 65)
        print(f"\n  {'Metric':<28} {'Score':<8}  {'Bar (0→1)':<22}  Status")
        print("  " + "-" * 63)

        scores_df = result.to_pandas()
        summary = {}

        all_metric_names = [
            "context_precision", "context_recall", "context_entity_recall",
            "answer_relevancy", "faithfulness", "noise_sensitivity",
        ]

        for metric in all_metric_names:
            if metric in scores_df.columns:
                avg = float(scores_df[metric].mean())
                summary[metric] = round(avg, 4) if not math.isnan(avg) else None
                _print_metric_summary(metric, avg)

        # ── Legend ──────────────────────────────────────────────────────────
        print("\n  ✓ GOOD = meets target  ⚠ FAIR = below target  ✗ BAD = needs attention")
        print("\n  Metric Targets:")
        for metric, guide in METRIC_GUIDE.items():
            if metric in summary:
                direction = "lower" if not guide["higher_is_better"] else "higher"
                print(f"    {metric:<28} target={guide['target']} ({direction} is better)")

        print(f"\n  Queries evaluated: {len(dataset_entries)}")
        print(f"  Evaluation mode:   {'retrieval-only' if retrieval_only else 'full pipeline'}")

        # ── Save results to JSON ───────────────────────────────────────────────
        output = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_mode": "retrieval_only" if retrieval_only else "full_pipeline",
            "pipeline_config": {
                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "chunk_size": 300,
                "chunk_overlap": 50,
                "top_k_per_query": TOP_K_PER_QUERY,
                "final_top_k": FINAL_TOP_K,
            },
            "num_queries": len(dataset_entries),
            "metrics_evaluated": list(summary.keys()),
            "summary_scores": summary,
            "metric_guide": METRIC_GUIDE,
            "per_query_scores": scores_df.to_dict(orient="records"),
        }

        with open(RESULTS_OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  Results saved to: {RESULTS_OUTPUT_PATH}")
        print("=" * 65)

    except ImportError as e:
        print(f"\n[WARNING] RAGAS import error: {e}")
        print("  Run: poetry add ragas")
    except Exception as e:
        print(f"\n[WARNING] RAGAS evaluation failed: {e}")
        print("  Make sure ChromaDB is populated and Ollama is running.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for the Blog-to-Podcast RAG pipeline"
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run only retrieval metrics (faster — skips full pipeline generation)",
    )
    args = parser.parse_args()
    evaluate(retrieval_only=args.retrieval_only)
