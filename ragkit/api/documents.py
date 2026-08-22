"""
Document API routes.
"""

from __future__ import annotations

from fastapi import APIRouter

from ragkit.models.document_info import DocumentInfo
from ragkit.services.document_service import DocumentService


router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
)


def create_document_router(
    document_service: DocumentService,
) -> APIRouter:
    """
    Create document API routes using the supplied service.
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