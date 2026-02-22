from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger
from app.config import settings

class VectorStoreManager:
    """Manages Vector DB operations"""

    def __init__(self, embedding_service: Embeddings):
        self.embedding_service = embedding_service
        self.persist_directory = settings.CHROMA_PERSIST_DIR

        try:
            self._vector_store = Chroma(
                collection_name = settings.CHROMA_COLLECTION_NAME,
                embedding_function = embedding_service,
                persist_directory = self.persist_directory                 
            )
            logger.info(f"Connected to ChromaDB: {settings.CHROMA_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Vector Store connection failed: {str(e)}")
            raise

    def upsert_documents(self, documents: List[Document]) -> bool:
        """Add or Update document in vector DB"""
        if not documents:
            logger.warning("Upsert invoked with empty document list.")
            return False
        
        try:
            logger.info(f"Indexing {len(documents)} chunks.")
            self._vector_store.add_documents(documents)
            logger.success("Indexing completed.")
            return True
        except Exception as e:
            logger.exception(f"Indexing failed: {str(e)}")
            return False

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Performs similarity search and returns top k results"""
        try:
            return self._vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Search failed for query '{query}' : {str(e)}")
            return []
            