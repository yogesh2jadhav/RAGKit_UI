from collections.abc import Iterable
from uuid import UUID, uuid4

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.search_result import SearchResult
from ragkit.services.document_service import DocumentService
from ragkit.vectorstores.vector_store import VectorStore


class FakeVectorStore(VectorStore):
    """
    Fake VectorStore used for DocumentService tests.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
    ) -> None:
        self._chunks = list(chunks)

    def add(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Embedding],
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> Iterable[SearchResult]:
        raise NotImplementedError

    def iter_chunks(
        self,
    ) -> Iterable[Chunk]:
        return iter(self._chunks)

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()


def create_chunk(
    document_id: UUID,
    filename: str,
    index: int,
) -> Chunk:
    """
    Create a test chunk.
    """

    content = f"Chunk {index}"

    return Chunk(
        id=uuid4(),
        document_id=document_id,
        index=index,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={
            "filename": filename,
            "uri": f"/documents/{filename}",
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        },
    )


def test_list_documents_returns_unique_documents():
    """
    Verify multiple chunks belonging to the same document
    produce one DocumentInfo.
    """

    document_id = uuid4()

    chunks = [
        create_chunk(
            document_id,
            "Yogesh Ashok 007.docx",
            0,
        ),
        create_chunk(
            document_id,
            "Yogesh Ashok 007.docx",
            1,
        ),
        create_chunk(
            document_id,
            "Yogesh Ashok 007.docx",
            2,
        ),
    ]

    service = DocumentService(
        vector_store=FakeVectorStore(chunks),
    )

    documents = service.list_documents()

    assert len(documents) == 1
    assert documents[0].id == document_id
    assert documents[0].filename == "Yogesh Ashok 007.docx"
    assert documents[0].chunk_count == 3


def test_list_documents_returns_multiple_documents():
    """
    Verify multiple document IDs produce multiple documents.
    """

    yogesh_id = uuid4()
    ashish_id = uuid4()

    chunks = [
        create_chunk(
            yogesh_id,
            "Yogesh Ashok 007.docx",
            0,
        ),
        create_chunk(
            yogesh_id,
            "Yogesh Ashok 007.docx",
            1,
        ),
        create_chunk(
            ashish_id,
            "Ashish Pawar (1).docx",
            0,
        ),
    ]

    service = DocumentService(
        vector_store=FakeVectorStore(chunks),
    )

    documents = service.list_documents()

    assert len(documents) == 2

    assert documents[0].filename == "Ashish Pawar (1).docx"
    assert documents[0].chunk_count == 1

    assert documents[1].filename == "Yogesh Ashok 007.docx"
    assert documents[1].chunk_count == 2


def test_list_documents_sorts_by_filename():
    """
    Verify documents are returned in filename order.
    """

    chunks = [
        create_chunk(
            uuid4(),
            "Yogesh Ashok 007.docx",
            0,
        ),
        create_chunk(
            uuid4(),
            "Ashish Pawar (1).docx",
            0,
        ),
        create_chunk(
            uuid4(),
            "Resume.docx",
            0,
        ),
    ]

    service = DocumentService(
        vector_store=FakeVectorStore(chunks),
    )

    documents = service.list_documents()

    filenames = [
        document.filename
        for document in documents
    ]

    assert filenames == [
        "Ashish Pawar (1).docx",
        "Resume.docx",
        "Yogesh Ashok 007.docx",
    ]


def test_list_documents_empty_store():
    """
    Verify an empty index returns an empty list.
    """

    service = DocumentService(
        vector_store=FakeVectorStore([]),
    )

    documents = service.list_documents()

    assert documents == []


def test_list_documents_falls_back_to_uri():
    """
    Verify filename can be recovered from URI when filename
    metadata is unavailable.
    """

    document_id = uuid4()

    chunk = Chunk(
        id=uuid4(),
        document_id=document_id,
        index=0,
        content="Test",
        start_offset=0,
        end_offset=4,
        metadata={
            "uri": "/documents/Resume.docx",
        },
    )

    service = DocumentService(
        vector_store=FakeVectorStore([chunk]),
    )

    documents = service.list_documents()

    assert len(documents) == 1
    assert documents[0].filename == "Resume.docx"