"""
Purpose
-------
Provides application-level access to indexed documents.

Responsibilities
----------------
- List indexed documents.
- Group chunks by document ID.
- Expose document information to applications.

Does NOT
--------
- Know about ChromaDB.
- Load source files.
- Generate embeddings.
- Perform RAG retrieval.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from uuid import UUID

from ragkit.models.document_info import DocumentInfo
from ragkit.models.chunk import Chunk
from ragkit.vectorstores.vector_store import VectorStore


class DocumentService:
    """
    Application service for indexed documents.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore,
    ) -> None:
        """
        Initialize the DocumentService.
        """

        self._vector_store = vector_store

    def list_documents(self) -> list[DocumentInfo]:
        """
        Return all documents currently represented in the index.

        Documents are reconstructed from indexed chunks.

        Returns
        -------
        list[DocumentInfo]
            Indexed documents sorted by filename.
        """

        documents: dict[UUID, dict[str, object]] = {}

        for chunk in self._vector_store.iter_chunks():

            document_id = chunk.document_id

            if document_id not in documents:
                documents[document_id] = {
                    "filename": self._get_filename(chunk),
                    "chunk_count": 0,
                }

            documents[document_id]["chunk_count"] = (
                int(documents[document_id]["chunk_count"]) + 1
            )

        result = [
            DocumentInfo(
                id=document_id,
                filename=str(info["filename"]),
                chunk_count=int(info["chunk_count"]),
            )
            for document_id, info in documents.items()
        ]

        result.sort(
            key=lambda document: document.filename.lower(),
        )

        return result

    @staticmethod
    def _get_filename(
        chunk: Chunk,
    ) -> str:
        """
        Get the filename stored in chunk metadata.
        """

        filename = chunk.metadata.get("filename")

        if filename:
            return str(filename)

        #
        # Fallback for older indexed data that may not
        # contain filename metadata.
        #
        uri = chunk.metadata.get("uri")

        if uri:
            return str(uri).rsplit("/", maxsplit=1)[-1]

        return str(chunk.document_id)