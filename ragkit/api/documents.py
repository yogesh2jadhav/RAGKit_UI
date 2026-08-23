"""
Purpose
-------
Expose document management APIs.

Responsibilities
----------------
- List indexed documents.
- Upload documents for indexing.
- Delete indexed documents.

Does NOT
--------
- Implement document indexing.
- Implement RAG retrieval.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, File, UploadFile

from ragkit.models.document_info import DocumentInfo
from ragkit.services.document_service import DocumentService


def create_document_router(
    document_service: DocumentService,
) -> APIRouter:
    """
    Create document management routes.
    """

    router = APIRouter(
        prefix="/api/documents",
        tags=["documents"],
    )

    @router.get(
        "/",
        response_model=list[DocumentInfo],
    )
    def list_documents() -> list[DocumentInfo]:
        """
        Return all indexed documents.
        """

        return document_service.list_documents()

    @router.post(
        "/upload",
        response_model=DocumentInfo,
    )
    async def upload_document(
        file: UploadFile = File(...),
    ) -> DocumentInfo:
        """
        Upload and index a .docx document.
        """

        if not file.filename:
            raise ValueError(
                "Uploaded file must have a filename."
            )

        content = await file.read()

        return document_service.upload_document(
            filename=file.filename,
            content=content,
        )

    @router.delete(
        "/{document_id}",
        status_code=204,
    )
    def delete_document(
        document_id: UUID,
    ) -> None:
        """
        Delete an indexed document and its source file.
        """

        document_service.delete_document(
            document_id=document_id,
        )

    return router