"""
Purpose
-------
HTTP endpoints for indexed documents.

Responsibilities
----------------
- Expose indexed documents through the REST API.
- Delegate document operations to DocumentService.

Does NOT
--------
- Access ChromaDB directly.
- Perform document indexing.
- Perform RAG retrieval.
"""

from __future__ import annotations

from fastapi import APIRouter

from ragkit.models.document_info import DocumentInfo
from ragkit.services.document_service import DocumentService


def create_document_router(
    document_service: DocumentService,
) -> APIRouter:
    """
    Create the document API router.
    """

    router = APIRouter(
        prefix="/api/documents",
        tags=["documents"],
    )

    @router.get(
        "",
        response_model=list[DocumentInfo],
    )
    def list_documents() -> list[DocumentInfo]:
        """
        Return all indexed documents.
        """

        return document_service.list_documents()

    return router