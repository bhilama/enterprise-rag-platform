import os
from loguru import logger
from langchain_community.document_loaders import Docx2txtLoader
from typing import List
from langchain_core.documents import Document

class DOCIngester:
    def __init__(self, file_path: str):
        logger.info(f"Initializing PDFIngestor for file {file_path}.")

        if not os.path.exists(file_path):
            logger.error(f"File validation failed {file_path}: does not exist.")
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path

    def load(self) -> List[Document]:
        #Loads Word document and adds custom metadata for tracking.
        logger.debug(f"Attempting to extract text from {self.file_path}.")
        try:
            loader = Docx2txtLoader(self.file_path)
            docs = loader.load()

            page_count = len(docs)
            logger.success(f"Successfully loaded {page_count} pages from {self.file_path}")

            #Enrich metadata for better filteing.
            for doc in docs:
                doc.metadata["file_type"] = "docx"
                doc.metadata["filename"] = os.path.basename(self.file_path)

            return docs
        except Exception as e:
            logger.critical(f"System failure while loading {self.file_path}: {str(e)}")
            return []