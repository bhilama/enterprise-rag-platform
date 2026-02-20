from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from loguru import logger

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        # Overlap is to make sure context is not lost at the edges.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators = ["\n\n", "\n", ".", " ", ""],
            add_start_index = True # adds a "tag" to the metadata showing exactly where in the original document this specific chunk started.
        )
        logger.info(f"Initialized DocumentChunker | Size: {self.chunk_size}, Overlap: {self.chunk_overlap}")

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents and provides structured logging of the process.
        """
        if not documents:
            logger.warning("Ingestion provided an empty document list. Skipping split.")
            return []
        
        logger.info(f"Starting split operation on {len(documents)} documents.")

        try:
            chunks = self.splitter.split_documents(documents)
            logger.success(f"Split complete: Created {len(chunks)} chunks.")

            # Log specific details for the first chunk to verify metadata
            if chunks:
                logger.debug(f"Chunk sample metadata: {chunks[0].metadata}")

            return chunks
        
        except Exception as e:
            # automatically captures the stack trace
            logger.exception(f"Unexpected error while document splitting: {str(e)}")
            raise