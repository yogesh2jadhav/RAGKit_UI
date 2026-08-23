"""
Purpose
-------
Provides application-level access to indexed documents.

Responsibilities
----------------
- List indexed documents.
- Group chunks by document ID.
- Upload and index documents.
- Delete indexed documents.
- Rebuild BM25 after indexing or deletion.

Does NOT
--------
- Know about ChromaDB.
- Load documents directly.
- Generate embeddings.
- Perform RAG retrieval.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.models.chunk import Chunk
from ragkit.models.document_info import DocumentInfo
from ragkit.sources.local_source import LocalSource
from ragkit.vectorstores.vector_store import VectorStore


class DocumentService:
    """
    Application service for indexed documents.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        document_indexer: DocumentIndexer | None = None,
        bm25_searcher: BM25Searcher | None = None,
    ) -> None:
        """
        Initialize the DocumentService.
        """

        self._vector_store = vector_store
        self._document_indexer = document_indexer
        self._bm25_searcher = bm25_searcher

    def list_documents(self) -> list[DocumentInfo]:
        """
        Return all documents currently represented in the index.

        Documents are reconstructed from indexed chunks.
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

    def upload_document(
        self,
        *,
        filename: str,
        content: bytes,
    ) -> DocumentInfo:
        """
        Save an uploaded .docx document permanently,
        index only the uploaded document, rebuild BM25,
        and return its indexed information.
        """

        if not filename:
            raise ValueError(
                "Filename is required."
            )

        if not filename.lower().endswith(".docx"):
            raise ValueError(
                "Only .docx files are supported."
            )

        if self._document_indexer is None:
            raise RuntimeError(
                "Document indexing is not configured."
            )

        if self._bm25_searcher is None:
            raise RuntimeError(
                "BM25 searcher is not configured."
            )

        project_root = Path(__file__).resolve().parents[2]

        documents_dir = project_root / "documents"

        documents_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_filename = Path(filename).name

        document_path = documents_dir / safe_filename

        #
        # Permanently store the original document.
        #
        document_path.write_bytes(content)

        #
        # LocalSource is directory based, so use a temporary
        # staging directory containing only this document.
        #
        with tempfile.TemporaryDirectory() as temp_dir:

            staged_path = Path(temp_dir) / safe_filename

            staged_path.write_bytes(content)

            source = LocalSource(
                directory=temp_dir,
            )

            result = self._document_indexer.index(
                source,
            )

        if result.documents != 1:
            raise RuntimeError(
                "Expected exactly one document to be indexed."
            )

        #
        # Rebuild the same BM25 instance used by RAGService.
        #
        self._bm25_searcher.rebuild()

        documents = self.list_documents()

        matching_documents = [
            document
            for document in documents
            if document.filename == safe_filename
        ]

        if not matching_documents:
            raise RuntimeError(
                "Uploaded document was indexed but could not "
                "be found in the document index."
            )

        return matching_documents[0]

    def delete_document(
        self,
        *,
        document_id: UUID,
    ) -> None:
        """
        Delete a document from the index and permanent storage.

        Steps
        -----
        1. Find the document in the index.
        2. Delete its chunks from the vector store.
        3. Delete the original .docx file.
        4. Rebuild BM25.
        """

        if self._bm25_searcher is None:
            raise RuntimeError(
                "BM25 searcher is not configured."
            )

        #
        # Find the document before deleting its chunks.
        #
        document = self._find_document(
            document_id,
        )

        if document is None:
            raise ValueError(
                f"Document '{document_id}' was not found."
            )

        #
        # Delete all chunks belonging to this document.
        #
        self._vector_store.delete_document(
            document_id,
        )

        #
        # Delete the original source document.
        #
        project_root = Path(__file__).resolve().parents[2]

        documents_dir = project_root / "documents"

        document_path = (
            documents_dir / document.filename
        )

        if document_path.exists():
            document_path.unlink()

        #
        # Rebuild the same BM25 instance used by RAGService.
        #
        self._bm25_searcher.rebuild()

    def _find_document(
        self,
        document_id: UUID,
    ) -> DocumentInfo | None:
        """
        Find an indexed document by ID.
        """

        for document in self.list_documents():
            if document.id == document_id:
                return document

        return None

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