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

    improvement_suggestions = state.get("improvement_suggestions", "")

    # Generate podcast script

    generation_input = {
        "context": context,
        "improvement_suggestions": improvement_suggestions,
    }
    script = script_generation_chain.invoke(generation_input)

    # Return only updated fields (partial state)
    return {
        "script": script,
        "generation_count": generation_count,
    }
