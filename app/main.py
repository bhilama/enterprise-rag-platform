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
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(raw_docs)
        logger.success(f"Processed {len(chunks)} total chunks.")

        if chunks:
            logger.info(f"Preview of first chunk: {chunks[0].page_content[:150]}")

    except Exception as e:
        logger.exception(f"Pipeline had issue during verification: {str(e)}")

if __name__ == "__main__":
    setup_logging()

    SAMPLE_PDF = "data/Test_PDF_File.pdf"

    if os.path.exists(SAMPLE_PDF):
        run_ingestion_verification(SAMPLE_PDF)
    else:
        logger.warning(f"Verification skipped: File not found at {SAMPLE_PDF}")


