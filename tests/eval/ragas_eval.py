"""
tests/eval/ragas_eval.py

PURPOSE:
    Standalone evaluation script for the Blog-to-Podcast RAG pipeline.
    Uses RAGAS (Retrieval-Augmented Generation Assessment) to measure:

    - Context Precision  : Of retrieved chunks, what % are actually relevant?
    - Context Recall     : Of relevant information, what % was retrieved?
    - Answer Relevancy   : Does the generated answer actually address the query?
    - Faithfulness       : Does the answer stick to retrieved context (no hallucination)?

USAGE:
    # Run evaluation (requires .chroma vector store to be populated first)
    poetry run python -m tests.eval.ragas_eval

    # Or directly
    poetry run python tests/eval/ragas_eval.py

REQUIREMENTS:
    - ChromaDB must be populated via ingestion (run ingestion first)
    - GROQ_API_KEY must be set in .env (used as the judge LLM — free tier)
    - Ollama must be running locally (for embedding queries)

HOW IT WORKS:
    1. Loads the golden dataset (5 hand-curated query + ground_truth pairs)
    2. For each query, runs our EXACT retrieval pipeline (retriever_chain + ChromaDB + reranker)
    3. Passes the (query, retrieved_contexts, ground_truths) to RAGAS
    4. RAGAS uses an LLM-as-judge (Groq llama-3.1-8b-instant) to score each metric
    5. Prints a summary table + saves results to tests/eval/ragas_results.json

WHY THIS APPROACH:
    LLM-as-judge evaluation is the industry standard for RAG systems where
    you don't have exact labeled answers. It scales better than human labeling
    and correlates well with human judgment at the retrieval quality level.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import graph modules
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


def run_retrieval_pipeline(query: str) -> list[str]:
    """
    Runs the exact same retrieval pipeline as the production node:
      1. retriever_chain  → generates multi-query strategy from query
      2. ChromaDB         → similarity search for each sub-query
      3. Deduplication    → removes duplicate chunks
      4. Reranking        → BGE cross-encoder scores and ranks chunks

    Returns:
        List of retrieved page_content strings (the contexts).
    """
    # Step 1: Generate retrieval strategy (multi-queries)
    # We create a minimal outline from the query for the strategy chain
    minimal_outline = {"episode_title": query, "key_talking_points": [{"topic": query}]}
    strategy = retriever_chain.invoke({"outline": json.dumps(minimal_outline, indent=2)})

    # Step 2: ChromaDB similarity search
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

    # Step 3: Deduplication
    seen = set()
    unique_docs = []
    for doc in all_docs:
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            unique_docs.append(doc)

    # Step 4: Reranking
    if not unique_docs:
        return []

    model = CrossEncoder(RERANKER_MODEL)
    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = model.predict(pairs)
    scored = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    final_docs = [doc for doc, _ in scored[:FINAL_TOP_K]]

    return [doc.page_content for doc in final_docs]


def evaluate():
    """
    Main evaluation loop.
    Loads the golden dataset, runs retrieval for each query,
    and computes RAGAS metrics using Groq as the judge LLM.
    """
    print("=" * 60)
    print("  RAGAS Evaluation — Blog-to-Podcast RAG Pipeline")
    print("=" * 60)

    # Load golden dataset
    with open(GOLDEN_DATASET_PATH) as f:
        golden = json.load(f)

    dataset_entries = golden["dataset"]
    print(f"\nLoaded {len(dataset_entries)} queries from golden dataset.\n")

    # ── Retrieve contexts for all queries ─────────────────────────────────────
    samples = []
    for i, entry in enumerate(dataset_entries, 1):
        query = entry["query"]
        ground_truths = entry["ground_truths"]

        print(f"[{i}/{len(dataset_entries)}] Retrieving for: '{query}'")
        try:
            contexts = run_retrieval_pipeline(query)
            print(f"    → Retrieved {len(contexts)} chunks")
        except Exception as e:
            print(f"    ⚠ Retrieval failed: {e}")
            contexts = []

        samples.append({
            "user_input": query,
            "retrieved_contexts": contexts,
            "reference": " ".join(ground_truths),  # RAGAS expects a single reference string
        })

    # ── Run RAGAS scoring ─────────────────────────────────────────────────────
    print("\nRunning RAGAS scoring (LLM-as-judge via Groq)...")
    print("This may take 1-2 minutes.\n")

    try:
        from ragas import evaluate as ragas_evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
        )
        from langchain_groq import ChatGroq
        from langchain_ollama import OllamaEmbeddings

        # Use Groq as the judge LLM (free tier)
        judge_llm = ChatGroq(model="llama-3.1-8b-instant")
        judge_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # Build RAGAS dataset
        ragas_samples = [
            SingleTurnSample(
                user_input=s["user_input"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s["reference"],
            )
            for s in samples
        ]
        eval_dataset = EvaluationDataset(samples=ragas_samples)

        # Run evaluation with available metrics
        # Note: AnswerRelevancy and Faithfulness require a 'response' field (generated answer).
        # We evaluate retrieval quality only here, which is the most meaningful metric
        # for tuning chunking and embedding strategies.
        metrics = [ContextPrecision(), ContextRecall()]
        result = ragas_evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
        )

        # ── Display Results ────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  RAGAS RESULTS")
        print("=" * 60)

        scores = result.to_pandas()
        summary = {}
        for metric in ["context_precision", "context_recall"]:
            if metric in scores.columns:
                avg = float(scores[metric].mean())
                summary[metric] = round(avg, 4)
                label = metric.replace("_", " ").title()
                bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
                print(f"  {label:<25} {avg:.4f}  [{bar}]")

        print("\n  Scores range from 0.0 (worst) to 1.0 (best)")
        print("  Target: Context Precision > 0.75, Context Recall > 0.70")

        # ── Save results to JSON ───────────────────────────────────────────────
        output = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "pipeline_config": {
                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "chunk_size": 300,
                "chunk_overlap": 50,
                "top_k_per_query": TOP_K_PER_QUERY,
                "final_top_k": FINAL_TOP_K,
            },
            "num_queries": len(dataset_entries),
            "summary_scores": summary,
            "per_query_scores": scores.to_dict(orient="records"),
        }

        with open(RESULTS_OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  Results saved to: {RESULTS_OUTPUT_PATH}")
        print("=" * 60)

    except ImportError as e:
        print(f"\n⚠ RAGAS import error: {e}")
        print("  Run: poetry add ragas")
    except Exception as e:
        print(f"\n⚠ RAGAS evaluation failed: {e}")
        print("  Make sure ChromaDB is populated and Ollama is running.")
        raise


if __name__ == "__main__":
    evaluate()
