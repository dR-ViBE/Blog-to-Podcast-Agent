from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def get_reranker(top_n: int = 5):
    """
    Initializes and returns a HuggingFace cross-encoder for reranking retrieved documents.
    Uses 'BAAI/bge-reranker-base'.
    """
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    return CrossEncoderReranker(model=model, top_n=top_n)
