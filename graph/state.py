from typing import List, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Docstring for GraphState

    param TypedDict: Description
    """
    url: str
    query: str
    documents: List[Document]
    script: str
    is_acceptable: bool
    script_evaluation: str
    generation_count: int
    max_generations: int
    audio_output: str
