import json
import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Set system console encoding to utf-8 to prevent cp1252 errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

CHROMA_COLLECTION = "blog_podcast_agent"
CHROMA_DIR = "./.chroma"
EMBEDDING_MODEL = "nomic-embed-text"

print("Connecting to ChromaDB...")
vectorstore = Chroma(
    collection_name=CHROMA_COLLECTION,
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=CHROMA_DIR,
)

# Fetch all chunks from the database
results = vectorstore.get()
ids = results.get("ids", [])
metadatas = results.get("metadatas", [])
documents = results.get("documents", [])

print(f"\nTotal chunks stored in database: {len(ids)}")

print("\n--- SAMPLE CHUNKS IN DATABASE ---")
# Show up to 3 sample chunks
for i in range(min(3, len(ids))):
    print(f"\nChunk {i+1}:")
    print(f"  ID: {ids[i]}")
    print(f"  Source: {metadatas[i].get('source', 'Unknown')}")
    print(f"  Page: {metadatas[i].get('page', 'Unknown')}")
    content_preview = documents[i][:300].replace('\n', ' ')
    print(f"  Content Preview: {content_preview}...")
