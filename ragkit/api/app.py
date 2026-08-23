"""
Purpose
-------
Create the RAGKit FastAPI application.

Responsibilities
----------------
- Create application dependencies.
- Register API routes.
- Serve the RAGKit web UI.

Does NOT
--------
- Implement RAG logic.
- Implement document indexing.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ragkit.api.chat import create_chat_router
from ragkit.api.documents import create_document_router
from ragkit.config.server_config import default_server_config
from ragkit.services.document_service import DocumentService
from ragkit.services.rag_service import RAGService
from ragkit.services.rag_service_factory import create_rag_service
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def create_app(
    *,
    document_service: DocumentService | None = None,
    rag_service: RAGService | None = None,
) -> FastAPI:
    """
    Create and configure the RAGKit FastAPI application.
    """

    app = FastAPI(
        title="RAGKit API",
        version="1.0.0",
    )

    # ------------------------------------------------------------------
    # Web UI
    # ------------------------------------------------------------------

    base_dir = Path(__file__).resolve().parents[2]

    static_dir = base_dir / "static"

    if not static_dir.exists():
        raise RuntimeError(
            f"Static directory does not exist: {static_dir}"
        )

    app.mount(
        "/static",
        StaticFiles(directory=static_dir),
        name="static",
    )

    @app.get("/")
    def serve_ui():
        """
        Serve the RAGKit web UI.
        """
        return FileResponse(
            static_dir / "index.html"
        )

    # ------------------------------------------------------------------
    # RAGKit dependencies
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # API routes
    # ------------------------------------------------------------------

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