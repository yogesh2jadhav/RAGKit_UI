"""
Purpose
-------
Create the RAGKit FastAPI application.

Responsibilities
----------------
- Create application dependencies.
- Register API routes.

Does NOT
--------
- Implement RAG logic.
- Implement document indexing.
"""

from __future__ import annotations
from ragkit.services.rag_service_factory import create_rag_service
from fastapi import FastAPI
from ragkit.api.chat import create_chat_router
from ragkit.api.documents import create_document_router
from ragkit.config.server_config import default_server_config
from ragkit.services.document_service import DocumentService
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore
from ragkit.services.rag_service import RAGService

def create_app(
    *,
    document_service: DocumentService | None = None,
    rag_service: RAGService | None = None,
) -> FastAPI:
    """
    Create and configure the RAGKit FastAPI application.
    """

    if document_service is None or rag_service is None:

        config = default_server_config()

        vector_store = ChromaVectorStore(
            path=config.vector_db_path,
            collection_name=config.collection_name,
        )

        if document_service is None:
            document_service = DocumentService(
                vector_store=vector_store,
            )

        if rag_service is None:
            rag_service = create_rag_service(
                vector_store=vector_store,
                embedding_model=config.embedding_model,
                llm_model=config.llm_model,
                top_k=config.retrieval_top_k,
            )

    app = FastAPI(
        title="RAGKit API",
        version="1.0.0",
    )

    app.include_router(
        create_document_router(
            document_service,
        )
    )

    app.include_router(
        create_chat_router(
            rag_service,
        )
    )

    return app


app = create_app()


app = create_app()