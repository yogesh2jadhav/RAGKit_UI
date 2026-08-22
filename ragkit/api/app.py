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

from fastapi import FastAPI

from ragkit.api.documents import create_document_router
from ragkit.config.server_config import default_server_config
from ragkit.services.document_service import DocumentService
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def create_app(
    document_service: DocumentService | None = None,
) -> FastAPI:
    """
    Create and configure the RAGKit FastAPI application.

    A DocumentService can be supplied by tests or by a
    higher-level application.
    """

    if document_service is None:

        config = default_server_config()

        vector_store = ChromaVectorStore(
            path=config.vector_db_path,
            collection_name=config.collection_name,
        )

        document_service = DocumentService(
            vector_store=vector_store,
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

    return app


app = create_app()


app = create_app()