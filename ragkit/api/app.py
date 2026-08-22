"""
RAGKit FastAPI application.
"""

from __future__ import annotations

from fastapi import FastAPI

from ragkit.services.document_service import DocumentService
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore

VECTOR_DB_PATH = "./data/chroma"
COLLECTION_NAME = "ragkit"


def create_app() -> FastAPI:
    """
    Create the RAGKit FastAPI application.
    """

    app = FastAPI(
        title="RAGKit API",
        version="1.0.0",
    )

    vector_store = ChromaVectorStore(
        path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    document_service = DocumentService(
        vector_store=vector_store,
    )

    from ragkit.api.documents import create_document_router

    app.include_router(
        create_document_router(
            document_service,
        )
    )

    return app


app = create_app()