from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from graph.graph import app

retriever = Chroma(
    collection_name="blog_podcast_agent_v2",  # v2: cosine distance
    persist_directory="./.chroma",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    collection_metadata={"hnsw:space": "cosine"},  # #8 Phase 1: cosine distance
).as_retriever()

docs = retriever.invoke("Summarize this blog as a podcast")

initial_state = {
    "documents": docs,
    "generation_count": 0,
    "max_generations": 3,
}

final_state = app.invoke(initial_state)

print(final_state)
