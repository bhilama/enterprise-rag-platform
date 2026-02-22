import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-DOC-INTELLIGENCE"

    # File ingestion settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

    # Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store settings
    CHROMA_PERSIST_DIR: str = "chroma_db"
    CHROMA_COLLECTION_NAME: str = "enterprise_docs"

    class Config:
        env_file = ".env"

settings = Settings()