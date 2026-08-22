from collections.abc import Iterable
from uuid import uuid4

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.search_result import SearchResult
from ragkit.services.rag_service import RAGService
from ragkit.services.rag_service_factory import create_rag_service


class FakeVectorStore:
    """
    Fake VectorStore used to verify factory wiring.
    """

    def iter_chunks(self) -> Iterable[Chunk]:
        """
        Return one fake chunk so BM25 can build its index.
        """

        yield Chunk(
            id=uuid4(),
            document_id=uuid4(),
            index=0,
            content="test document content",
            start_offset=0,
            end_offset=20,
            metadata={
                "filename": "test.docx",
            },
        )

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

    def count(self) -> int:
        return 0

    def clear(self) -> None:
        raise NotImplementedError


def test_create_rag_service():
    """
    Verify the factory creates a RAGService.
    """

    service = create_rag_service(
        vector_store=FakeVectorStore(),
        embedding_model="test-embedding",
        llm_model="test-llm",
        top_k=5,
    )

    assert isinstance(
        service,
        RAGService,
    )