import os
from loguru import logger
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document

class PDFIngestor:
    def __init__(self, file_path: str):
        logger.info(f"Initializing PDFIngestor for file {file_path}.")

        if not os.path.exists(file_path):
            logger.error(f"File validation failed {file_path}: does not exist.")
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path

    def load(self) -> List[Document]:
        #loads PDF and adds custom metadata for tracking.
        logger.debug(f"Attempting to extract text from {self.file_path}.")
        try:
            loader = PyPDFLoader(self.file_path)
            docs = loader.load()

            page_count = len(docs)
            logger.success(f"Successfully loaded {page_count} pages from {self.file_path}")

            #Enrich metadata for better filtering.
            for doc in docs:
                doc.metadata["file_type"] = "pdf"
                doc.metadata["filename"] = os.path.basename(self.file_path)
            
            return docs
        except Exception as e:
            logger.critical(f"System failure while loading {self.file_path}: {str(e)}")
            return []
