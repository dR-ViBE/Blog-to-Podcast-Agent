from graph.state import GraphState
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import Dict, List
from langchain_core.documents import Document


def retrieve_blog_chunks(state: GraphState) -> Dict:
    query = state.get("query", "")

    vectorstore = Chroma(collection_name="blog_podcast_agent",
                         embedding_function=OllamaEmbeddings(
                             model="nomic-embed-text"),
                         persist_directory="./.chroma")

    retrieved_docs: List[Document] = vectorstore.similarity_search(
        query=query, k=6)

    return {"documents": retrieved_docs}
