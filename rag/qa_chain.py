import os
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from app.config import settings

class QAChainManager:
    """Manages LLM orchestration using a custom internal API gateway (llmapi.ai)"""

    def __init__(self, retriever):
        # Initialize the LLM with custom Base URL
        logger.info("Initializing QA chain with LLM API")   
        try:
            self.llm =  ChatOpenAI(
                model = "gpt-4o",
                temperature=0,
                openai_api_key = os.getenv("LLM_API_KEY"),
                openai_api_base = "https://internal.llmapi.ai/v1",
                max_retries = 2
            )

            # Define System prompt logic
            system_prompt = (
                "You are a professional corporate assistant. Use the following pieces of "
                "retrieved context to answer the user's question. If the answer is not "
                "in the context, say that you do not have enough information. "
                "\n\nContext:\n{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
        
            # Build retrieval Chain
            combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

            logger.success("QA Chain connected to internal LLM API.")

        except Exception as e:
            logger.critical(f"Failed to connect to internal LLM API: {str(e)}")
            raise

    def ask(self, user_query:str) -> dict:
        """Process query through RAG pipeline"""

        try:
            response = self.rag_chain.invoke({"input": user_query})
            return {
                "answer": response["answer"],
                "source": response["context"]
            }
        except Exception as e:
            logger.error(f"Error during RAG processing: {str(e)}")
            return {"answer": "Error: Unable to reach internal LLM service.", "sources": []}

