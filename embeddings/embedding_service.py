from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from loguru import logger
from app.config import settings

class BaseEmbeddingService(ABC):
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        pass

class LocalEmbeddingsService(BaseEmbeddingService):
    """Implementation for local HuggingFace embeddings."""

    def __init__(self):
        try:
            logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name = settings.EMBEDDING_MODEL,
                model_kwargs = {'device':'cpu'}
            )
            logger.success("Embedding service initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize embedding service: {str(e)}")
            raise
    
    
    def get_embeddings(self) -> Embeddings:
        return self._embeddings