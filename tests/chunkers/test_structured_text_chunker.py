"""
Tests for StructuredTextChunker.
"""

from uuid import uuid4

import pytest

from ragkit.chunkers.structured_text_chunker import (
    StructuredTextChunker,
)
from ragkit.models.document import Document


def create_document(content: str) -> Document:
    """
    Create a test Document.
    """

    return Document(
        id=uuid4(),
        content=content,
        metadata={
            "filename": "test.docx",
        },
    )


def test_keeps_logical_blocks_together() -> None:
    """
    Related text blocks should remain together when
    they fit within the configured chunk size.
    """

    document = create_document(
        "Key Skills\n\n"
        "Java, Spark, Scala, Databricks\n\n"
        "Work Experience\n\n"
        "Solution Architect"
    )

    chunker = StructuredTextChunker(
        chunk_size=200,
        chunk_overlap=20,
    )

    chunks = list(
        chunker.chunk(document)
    )

    assert len(chunks) == 1

    assert "Key Skills" in chunks[0].content
    assert "Java, Spark, Scala, Databricks" in chunks[0].content
    assert "Work Experience" in chunks[0].content
    assert "Solution Architect" in chunks[0].content


def test_splits_when_chunk_size_is_exceeded() -> None:
    """
    Logical blocks should be distributed across multiple
    chunks when the configured size is exceeded.
    """

    document = create_document(
        "A" * 40
        + "\n\n"
        + "B" * 40
        + "\n\n"
        + "C" * 40
    )

    chunker = StructuredTextChunker(
        chunk_size=90,
        chunk_overlap=10,
    )

    chunks = list(
        chunker.chunk(document)
    )

    assert len(chunks) == 2

    assert "A" * 40 in chunks[0].content
    assert "B" * 40 in chunks[0].content

    assert "C" * 40 in chunks[1].content


def test_preserves_document_id() -> None:
    """
    Every generated chunk should reference the source document.
    """

    document = create_document(
        "Apache Spark\n\n"
        "Distributed processing"
    )

    chunker = StructuredTextChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = list(
        chunker.chunk(document)
    )

    assert chunks

    for chunk in chunks:
        assert chunk.document_id == document.id


def test_preserves_metadata() -> None:
    """
    Document metadata should be copied to every chunk.
    """

    document = create_document(
        "Apache Spark"
    )

    chunker = StructuredTextChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = list(
        chunker.chunk(document)
    )

    assert chunks[0].metadata == document.metadata


def test_chunk_indexes_are_sequential() -> None:
    """
    Chunk indexes should start at zero and increase sequentially.
    """

    document = create_document(
        "A" * 40
        + "\n\n"
        + "B" * 40
        + "\n\n"
        + "C" * 40
    )

    chunker = StructuredTextChunker(
        chunk_size=60,
        chunk_overlap=10,
    )

    chunks = list(
        chunker.chunk(document)
    )

    assert [
        chunk.index
        for chunk in chunks
    ] == list(range(len(chunks)))


def test_empty_document_produces_no_chunks() -> None:
    """
    Empty documents should produce no chunks.
    """

    document = create_document("")

    chunker = StructuredTextChunker()

    chunks = list(
        chunker.chunk(document)
    )

    assert chunks == []


def test_rejects_invalid_chunk_size() -> None:
    """
    chunk_size must be positive.
    """

    with pytest.raises(ValueError):
        StructuredTextChunker(
            chunk_size=0,
        )


def test_rejects_negative_overlap() -> None:
    """
    chunk_overlap cannot be negative.
    """

    with pytest.raises(ValueError):
        StructuredTextChunker(
            chunk_overlap=-1,
        )


def test_rejects_overlap_equal_to_chunk_size() -> None:
    """
    Overlap must be smaller than chunk_size.
    """

    with pytest.raises(ValueError):
        StructuredTextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_large_block_uses_character_fallback() -> None:
    """
    A logical block larger than chunk_size should still
    be split rather than creating an oversized chunk.
    """

    document = create_document(
        "A" * 250
    )

    chunker = StructuredTextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = list(
        chunker.chunk(document)
    )

    assert len(chunks) == 3

    assert len(chunks[0].content) == 100
    assert len(chunks[1].content) == 100
    assert len(chunks[2].content) == 90