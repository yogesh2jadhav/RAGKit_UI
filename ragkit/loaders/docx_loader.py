"""
Purpose
-------
Load Microsoft Word (.docx) documents into RagKit Documents.

Responsibilities
----------------
- Support .docx files.
- Extract paragraph text.
- Extract table text.
- Preserve the order of paragraphs and tables where possible.
- Create a RagKit Document.
- Preserve source metadata.

Does NOT
--------
- Chunk documents.
- Generate embeddings.
- Store vectors.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from docx import Document as WordDocument

from ragkit.loaders.loader import Loader
from ragkit.models.document import Document
from ragkit.models.source_document import SourceDocument


class DocxLoader(Loader):
    """
    Loader for Microsoft Word .docx documents.
    """

    @staticmethod
    def supports(source: SourceDocument) -> bool:
        """
        Return True when the source is a DOCX document.

        Extension matching is case-insensitive.
        """

        return Path(source.uri).suffix.lower() == ".docx"

    def load(self, source: SourceDocument) -> Document:
        """
        Load a DOCX document into a RagKit Document.

        Paragraph text and table-cell text are extracted.
        Empty paragraphs and empty table cells are ignored.
        """

        word_document = WordDocument(source.uri)

        content_parts: list[str] = []

        #
        # Extract normal paragraphs.
        #
        for paragraph in word_document.paragraphs:

            text = paragraph.text.strip()

            if text:
                content_parts.append(text)

        #
        # Extract table content.
        #
        for table in word_document.tables:

            for row in table.rows:

                row_cells: list[str] = []

                for cell in row.cells:

                    text = cell.text.strip()

                    if text:
                        row_cells.append(text)

                if row_cells:
                    content_parts.append(" | ".join(row_cells))

        content = "\n".join(content_parts)

        #
        # Start with metadata supplied by the source.
        #
        metadata = dict(source.metadata)

        #
        # Add standard metadata provided by the loader.
        #
        metadata.update(
            {
                "uri": source.uri,
                "mime_type": source.mime_type,
                "filename": Path(source.uri).name,
            }
        )

        #
        # Every RagKit Document requires a unique ID.
        #
        return Document(
            id=uuid4(),
            content=content,
            metadata=metadata,
        )