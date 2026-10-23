import json
import os
import sys
from pathlib import Path

# Set system console encoding to utf-8 to prevent cp1252 errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables before importing graph packages to ensure API keys are set
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from graph.chains.retriever_chain import retriever_chain

# Settings
CHROMA_COLLECTION = "blog_podcast_agent"
CHROMA_DIR = "./.chroma"
EMBEDDING_MODEL = "nomic-embed-text"
RERANKER_MODEL = "BAAI/bge-reranker-base"
TOP_K_PER_QUERY = 4
FINAL_TOP_K = 6

query = "What are the challenges associated with using traditional trajectory planners on complex hydraulic excavators?"

print("==============================================================")
print(f"ORIGINAL QUERY: '{query}'")
print("==============================================================")

# Step 1: Generate search queries
search_queries = []
try:
    print("\n[Step 1] Planning outline using planner_chain...")
    from graph.chains.planner_chain import planner_chain
    outline = planner_chain.invoke({"query": query})
    print(f"  Generated Outline: '{outline.episode_title}'")

    print("\nGenerating search queries using retriever_chain...")
    outline_dict = outline.dict()
    # Limit talking points to prevent LLM schema errors
    outline_dict["key_talking_points"] = outline_dict["key_talking_points"][:5]
    outline_json = json.dumps(outline_dict, indent=2)
    strategy = retriever_chain.invoke({"outline": outline_json})
    search_queries = strategy.queries if strategy.queries else [query]
except Exception as e:
    print(f"\n[WARNING] retriever_chain failed: {e}. Falling back to default search queries...")
    # fallback search queries based on the original query
    search_queries = [
        "What are the challenges of traditional trajectory planners on complex hydraulic excavators?",
        "Why are hydraulic excavators difficult to model analytically?",
        "Excavator dynamics machine-specific nonlinear trajectory planning challenges"
    ]

print(f"\nUsing {len(search_queries)} search queries for ChromaDB:")
for i, q in enumerate(search_queries, 1):
    print(f"  {i}. '{q}'")

# Step 2: Fetch raw chunks from ChromaDB
print("\n[Step 2] Fetching chunks from ChromaDB similarity search...")
vectorstore = Chroma(
    collection_name=CHROMA_COLLECTION,
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=CHROMA_DIR,
)

all_docs = []
for q in search_queries:
    docs = vectorstore.similarity_search(query=q, k=TOP_K_PER_QUERY)
    print(f"  Query '{q}' -> Found {len(docs)} chunks")
    all_docs.extend(docs)

# Step 3: Deduplicate chunks
print(f"\n[Step 3] Deduplicating {len(all_docs)} total chunks retrieved...")
seen = set()
unique_docs = []
for doc in all_docs:
    h = hash(doc.page_content)
    if h not in seen:
        seen.add(h)
        unique_docs.append(doc)
print(f"  Deduplicated down to {len(unique_docs)} unique chunks.")

# Step 4: Rerank using CrossEncoder
print("\n[Step 4] Scoring & Reranking unique chunks using CrossEncoder...")
model = CrossEncoder(RERANKER_MODEL)
pairs = [[query, doc.page_content] for doc in unique_docs]
scores = model.predict(pairs)

scored = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)

print(f"\n--- RERANKED CHUNKS (Top {FINAL_TOP_K}) ---")
for idx, (doc, score) in enumerate(scored[:FINAL_TOP_K], 1):
    print(f"\n[{idx}] Score: {score:.4f} | Page: {doc.metadata.get('page', 'Unknown')}")
    print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
    snippet = doc.page_content[:400].replace('\n', ' ')
    print(f"    Content: {snippet}...")
