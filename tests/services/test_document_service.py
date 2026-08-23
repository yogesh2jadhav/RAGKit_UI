from collections.abc import Iterable
from pathlib import Path
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
        self.deleted_document_ids: list[UUID] = []

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

    def delete_document(
        self,
        document_id: UUID,
    ) -> None:
        """
        Delete all chunks belonging to a document.
        """

        self.deleted_document_ids.append(
            document_id,
        )

        self._chunks = [
            chunk
            for chunk in self._chunks
            if chunk.document_id != document_id
        ]


class FakeBM25Searcher:
    """
    Fake BM25 searcher used for DocumentService tests.
    """

    def __init__(self) -> None:
        self.rebuild_count = 0

    def rebuild(self) -> None:
        """
        Record BM25 rebuild calls.
        """

        self.rebuild_count += 1


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
            "mime_type": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
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


def test_delete_document_removes_vector_chunks():
    """
    Verify deleting a document removes only its chunks
    from the vector store.
    """

    document_id = uuid4()
    other_document_id = uuid4()

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
            other_document_id,
            "Ashish Pawar (1).docx",
            0,
        ),
    ]

    vector_store = FakeVectorStore(chunks)

    bm25_searcher = FakeBM25Searcher()

    service = DocumentService(
        vector_store=vector_store,
        bm25_searcher=bm25_searcher,
    )

    service.delete_document(
        document_id=document_id,
    )

    assert vector_store.deleted_document_ids == [
        document_id,
    ]

    remaining_chunks = list(
        vector_store.iter_chunks(),
    )

    assert len(remaining_chunks) == 1
    assert remaining_chunks[0].document_id == other_document_id


def test_delete_document_deletes_physical_file(
    tmp_path: Path,
    monkeypatch,
):
    """
    Verify deleting a document removes its original
    physical .docx file.
    """

    document_id = uuid4()

    chunk = create_chunk(
        document_id,
        "Yogesh Ashok 007.docx",
        0,
    )

    vector_store = FakeVectorStore(
        [chunk],
    )

    bm25_searcher = FakeBM25Searcher()

    service = DocumentService(
        vector_store=vector_store,
        bm25_searcher=bm25_searcher,
    )

    documents_dir = tmp_path / "documents"

    documents_dir.mkdir()

    document_path = (
        documents_dir / "Yogesh Ashok 007.docx"
    )

    document_path.write_bytes(
        b"test document",
    )

    #
    # Patch the Path used by DocumentService so that
    # its project root resolves to tmp_path.
    #
    original_resolve = Path.resolve

    def fake_resolve(path: Path) -> Path:
        if path.name == "document_service.py":
            return (
                tmp_path
                / "ragkit"
                / "services"
                / "document_service.py"
            )

        return original_resolve(path)

    monkeypatch.setattr(
        Path,
        "resolve",
        fake_resolve,
    )

    assert document_path.exists()

    service.delete_document(
        document_id=document_id,
    )

    assert not document_path.exists()

def test_delete_document_rebuilds_bm25():
    """
    Verify deleting a document rebuilds BM25.
    """

    document_id = uuid4()

    chunks = [
        create_chunk(
            document_id,
            "Yogesh Ashok 007.docx",
            0,
        ),
    ]

    vector_store = FakeVectorStore(chunks)

    bm25_searcher = FakeBM25Searcher()

    service = DocumentService(
        vector_store=vector_store,
        bm25_searcher=bm25_searcher,
    )

    service.delete_document(
        document_id=document_id,
    )

    assert bm25_searcher.rebuild_count == 1


def test_delete_document_rejects_unknown_document():
    """
    Verify deleting an unknown document raises ValueError.
    """

    vector_store = FakeVectorStore([])

    bm25_searcher = FakeBM25Searcher()

    service = DocumentService(
        vector_store=vector_store,
        bm25_searcher=bm25_searcher,
    )

    unknown_document_id = uuid4()

    try:
        service.delete_document(
            document_id=unknown_document_id,
        )
    except ValueError as exc:
        assert str(unknown_document_id) in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for unknown document."
        )