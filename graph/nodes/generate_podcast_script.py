from typing import Dict
from langchain_core.documents import Document
from graph.chains.podcast_script_chain import script_generation_chain
from graph.state import GraphState


def generate_podcast_script(state: GraphState) -> Dict:

    print("---Generate Podcast Script---")
    # Read documents from state
    documents = state.get("documents", [])

    # Combine all document contents into a single context string
    context = "\n\n".join(
        doc.page_content for doc in documents if isinstance(doc, Document)
    )

    # Read and increment generation count
    generation_count = state.get("generation_count", 0) + 1

    # Generate podcast script
    script = script_generation_chain.invoke({"context": context})

    # Return only updated fields (partial state)
    return {
        "script": script,
        "generation_count": generation_count,
    }
