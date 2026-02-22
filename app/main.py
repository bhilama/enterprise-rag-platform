import sys
import os
from loguru import logger

# Setup Loguru
def setup_logging():
    # Handle console and file logging.
    logger.remove() # Remove default handler

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # Log file
    logger.add(
        "logs/pipeline_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",
        level="DEBUG",
        compression="zip"
    )

# Import local modules now
from ingestion.pdf_loader import PDFIngestor
from ingestion.chunker import DocumentChunker
from embeddings.embedding_service import LocalEmbeddingsService
from vector_store.chroma_store import VectorStoreManager
from app.config import settings

def run_ingestion_verification(file_path: str):
    logger.info(f"Starting verification for {os.path.basename(file_path)}.")

    try:
        logger.info("Loading a file.")
        loader = PDFIngestor(file_path)
        raw_docs = loader.load()

        if not raw_docs:
            logger.error("No pages were loaded. Aborting pipeline.")
            return
        
        logger.info("Breaking documents into chunks.")
        chunker = DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        chunks = chunker.split(raw_docs)
        logger.success(f"Processed {len(chunks)} total chunks.")

        if chunks:
            logger.info(f"Preview of first chunk: {chunks[0].page_content[:150]}")

        # Embedding service initialization 
        logger.info("Initializing Embedding service.")
        embedding_service = LocalEmbeddingsService().get_embeddings()

        # Vector Store initialization and indexing
        logger.info("Initializing Vector store and indexing.")
        vector_manager = VectorStoreManager(embedding_service=embedding_service)

        # Add chunks to ChromaDb
        success = vector_manager.upsert_documents(chunks)

        if success:
            logger.info("Testing similarity search")
            test_query = "What is this document about?"
            test_results = vector_manager.similarity_search(test_query, k=2)

            if test_results:
                logger.success(f"Search Successful. Found {len(test_results)} result contexts")

                for i, result in enumerate(test_results):
                    logger.info(f"Result {i+1} (Page {result.metadata.get('page', 'N/A')}): {result.page_content[:500]}...")
            else:
                logger.warning("Search returned no results. Check embeddings and indexing.")
    except Exception as e:
        logger.exception(f"Pipeline had issue during verification: {str(e)}")

if __name__ == "__main__":
    setup_logging()

    SAMPLE_PDF = "data/Test_PDF_File.pdf"

    if os.path.exists(SAMPLE_PDF):
        run_ingestion_verification(SAMPLE_PDF)
    else:
        logger.warning(f"Verification skipped: File not found at {SAMPLE_PDF}")


