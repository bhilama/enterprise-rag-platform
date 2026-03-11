from loguru import logger
from typing import List
from langchain_core.documents import Document
from vector_store.chroma_store import VectorStoreManager

class DocumentRetriever:
    """Encapsulates search logic and allow for future re-ranking"""

    def __init__(self, vector_manager: VectorStoreManager, search_k: int = 3):
        self.retriever = vector_manager._vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": search_k}
        )
        logger.info(f"DocumentRetriever initialized with k = {search_k}")

    def get_context(self, query: str) -> List[Document]:
        """Fetched relevant documents chunk based on the query."""
        logger.info(f"Retrieving context for query: '{query}'")

        try:
            docs = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} relevant chunks.")
            return docs
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            return []