from uuid import uuid4

import pytest
from pytest import raises

from ragkit.exceptions.vector_store_error import VectorStoreError
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def create_chunk(content: str, index: int = 0) -> Chunk:
    """
    Create a Chunk for testing.
    """
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=index,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={
            "source": "unit-test",
        },
    )


def create_embedding(chunk: Chunk, vector: list[float]) -> Embedding:
    """
    Create an Embedding for testing.
    """
    return Embedding(
        chunk_id=chunk.id,
        model="unit-test-model",
        vector=vector,
    )


def test_store_embedding(tmp_path):
    """
    Verify a chunk and embedding are stored.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = create_chunk("Apache Spark")

    embedding = create_embedding(
        chunk,
        [0.1, 0.2, 0.3],
    )

    store.add(
        chunks=[chunk],
        embeddings=[embedding],
    )

    assert store.count() == 1


def test_missing_embedding_for_chunk(tmp_path):
    """
    Verify missing embeddings are rejected.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = create_chunk("Apache Spark")

    embedding = Embedding(
        chunk_id=uuid4(),
        model="unit-test-model",
        vector=[0.1, 0.2],
    )

    with raises(VectorStoreError):
        store.add(
            chunks=[chunk],
            embeddings=[embedding],
        )


def test_store_embeddings_in_any_order(tmp_path):
    """
    Verify ordering does not matter.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk1 = create_chunk("Chunk 1", 0)
    chunk2 = create_chunk("Chunk 2", 1)

    embedding1 = create_embedding(
        chunk1,
        [0.1, 0.2],
    )

    embedding2 = create_embedding(
        chunk2,
        [0.3, 0.4],
    )

    store.add(
        chunks=[chunk1, chunk2],
        embeddings=[embedding2, embedding1],
    )

    assert store.count() == 2


def test_search_empty_database(tmp_path):
    """
    Searching an empty database should return no results.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    query = QueryEmbedding(
        model="unit-test-model",
        vector=[0.1, 0.2],
    )

    results = list(
        store.search(
            query_embedding=query,
            top_k=5,
        )
    )

    assert results == []


def test_search_returns_chunk(tmp_path):
    """
    Verify a stored chunk can be retrieved.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = create_chunk("Apache Spark")

    embedding = create_embedding(
        chunk,
        [0.1, 0.2, 0.3],
    )

    store.add(
        chunks=[chunk],
        embeddings=[embedding],
    )

    query = QueryEmbedding(
        model="unit-test-model",
        vector=[0.1, 0.2, 0.3],
    )

    results = list(
        store.search(
            query_embedding=query,
            top_k=1,
        )
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk.content == chunk.content
    assert result.chunk.document_id == chunk.document_id
    assert result.chunk.index == chunk.index
    assert result.chunk.metadata == chunk.metadata


def test_search_top_k(tmp_path):
    """
    Verify top_k limits the number of returned results.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunks = []

    embeddings = []

    for i in range(5):

        chunk = create_chunk(
            f"Chunk {i}",
            i,
        )

        chunks.append(chunk)

        embeddings.append(
            create_embedding(
                chunk,
                [float(i), float(i + 1)],
            )
        )

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    query = QueryEmbedding(
        model="unit-test-model",
        vector=[0.0, 1.0],
    )

    results = list(
        store.search(
            query_embedding=query,
            top_k=2,
        )
    )

    assert len(results) == 2

    def test_search_empty_database(tmp_path):
        """
        Verify searching an empty vector store returns no results.
        """

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        query = QueryEmbedding(
            model="unit-test-model",
            vector=[0.1, 0.2, 0.3],
        )

        results = list(
            store.search(
                query_embedding=query,
                top_k=5,
            )
        )

        assert results == []

    def test_search_returns_chunk(tmp_path):
        """
        Verify a stored chunk can be retrieved.
        """

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        chunk = create_chunk("Apache Spark")

        embedding = create_embedding(
            chunk,
            [0.1, 0.2, 0.3],
        )

        store.add(
            chunks=[chunk],
            embeddings=[embedding],
        )

        query = QueryEmbedding(
            model="unit-test-model",
            vector=[0.1, 0.2, 0.3],
        )

        results = list(
            store.search(
                query_embedding=query,
                top_k=1,
            )
        )

        assert len(results) == 1

        result = results[0]

        assert result.chunk.id == chunk.id
        assert result.chunk.document_id == chunk.document_id
        assert result.chunk.index == chunk.index
        assert result.chunk.content == chunk.content
        assert result.chunk.start_offset == chunk.start_offset
        assert result.chunk.end_offset == chunk.end_offset
        assert result.chunk.metadata == chunk.metadata

    def test_search_top_k(tmp_path):
        """
        Verify search respects the requested top_k value.
        """

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        chunks = []
        embeddings = []

        for i in range(5):
            chunk = create_chunk(
                f"Chunk {i}",
                index=i,
            )

            chunks.append(chunk)

            embeddings.append(
                create_embedding(
                    chunk,
                    [float(i), float(i + 1)],
                )
            )

        store.add(
            chunks=chunks,
            embeddings=embeddings,
        )

        query = QueryEmbedding(
            model="unit-test-model",
            vector=[0.0, 1.0],
        )

        results = list(
            store.search(
                query_embedding=query,
                top_k=2,
            )
        )

        assert len(results) == 2

    def test_search_invalid_top_k(tmp_path):
        """
        Verify search rejects invalid top_k values.
        """

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        query = QueryEmbedding(
            model="unit-test-model",
            vector=[0.1, 0.2],
        )

        with pytest.raises(ValueError):
            list(
                store.search(
                    query_embedding=query,
                    top_k=0,
                )
            )

    def test_search_preserves_user_metadata(tmp_path):
        """
        Verify that user-defined metadata is preserved after
        storing and retrieving a Chunk.

        Internal RAGKit metadata must not leak into the
        reconstructed Chunk.metadata.
        """

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            index=0,
            content="Apache Spark",
            start_offset=0,
            end_offset=12,
            metadata={
                "author": "Yogesh",
                "language": "en",
                "document_id": "user-document-id",
                "model": "user-model",
            },
        )

        embedding = Embedding(
            chunk_id=chunk.id,
            model="unit-test-model",
            vector=[0.1, 0.2, 0.3],
        )

        store.add(
            chunks=[chunk],
            embeddings=[embedding],
        )

        query = QueryEmbedding(
            model="unit-test-model",
            vector=[0.1, 0.2, 0.3],
        )

        results = list(
            store.search(
                query_embedding=query,
                top_k=1,
            )
        )

        assert len(results) == 1

        retrieved_chunk = results[0].chunk

        #
        # Core Chunk fields must be reconstructed correctly.
        #
        assert retrieved_chunk.id == chunk.id
        assert retrieved_chunk.document_id == chunk.document_id
        assert retrieved_chunk.index == chunk.index
        assert retrieved_chunk.content == chunk.content
        assert retrieved_chunk.start_offset == chunk.start_offset
        assert retrieved_chunk.end_offset == chunk.end_offset

        #
        # User metadata must survive the round trip.
        #
        assert retrieved_chunk.metadata["author"] == "Yogesh"
        assert retrieved_chunk.metadata["language"] == "en"
        assert retrieved_chunk.metadata["document_id"] == "user-document-id"
        assert retrieved_chunk.metadata["model"] == "user-model"

        #
        # Internal RAGKit metadata must not leak into user metadata.
        #
        assert "_ragkit_document_id" not in retrieved_chunk.metadata
        assert "_ragkit_chunk_index" not in retrieved_chunk.metadata
        assert "_ragkit_start_offset" not in retrieved_chunk.metadata
        assert "_ragkit_end_offset" not in retrieved_chunk.metadata
        assert "_ragkit_model" not in retrieved_chunk.metadata

    def test_clear_removes_all_embeddings(tmp_path):
        """
        Verify clear removes all stored embeddings.
        """

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        chunk1 = create_chunk("Chunk 1")
        chunk2 = create_chunk("Chunk 2", 1)

        embedding1 = create_embedding(
            chunk1,
            [0.1, 0.2, 0.3],
        )

        embedding2 = create_embedding(
            chunk2,
            [0.4, 0.5, 0.6],
        )

        store.add(
            chunks=[chunk1, chunk2],
            embeddings=[embedding1, embedding2],
        )

        assert store.count() == 2

        store.clear()

        assert store.count() == 0

    def test_clear_empty_store(tmp_path):
        """
        Verify clear works when the vector store is already empty.
        """

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        assert store.count() == 0

        store.clear()

        assert store.count() == 0