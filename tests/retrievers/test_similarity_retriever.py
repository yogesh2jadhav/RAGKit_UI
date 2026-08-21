from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from ragkit.config.retrieval_config import RetrievalConfig
from ragkit.embeddings.embedder import Embedder
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.search_result import SearchResult
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.vectorstores.vector_store import VectorStore


class FakeEmbedder(Embedder):
    """
    Fake Embedder used for unit testing.
    """

    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:
        return []

    def embed_query(
        self,
        query: str,
    ) -> QueryEmbedding:
        return QueryEmbedding(
            model="unit-test-model",
            vector=[0.1, 0.2, 0.3],
        )


class FakeVectorStore(VectorStore):
    """
    Fake VectorStore used for unit testing.
    """

    def __init__(self) -> None:
        self.last_top_k = None
        self.last_query_embedding = None
        self.last_filters = None

    def add(
        self,
        chunks,
        embeddings,
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ):
        self.last_query_embedding = query_embedding
        self.last_top_k = top_k
        self.last_filters = filters

        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            index=0,
            content="Apache Spark",
            start_offset=0,
            end_offset=12,
            metadata={
                "source": "unit-test",
            },
        )

        yield SearchResult(
            chunk=chunk,
            score=0.123,
        )

    def clear(self) -> None:
        """
        Clear the fake vector store.
        """
        self.last_top_k = None
        self.last_query_embedding = None
        self.last_filters = None

    def count(self) -> int:
        return 0

    def iter_chunks(
        self,
    ) -> Iterable[Chunk]:
        """
        Fake implementation used only for testing.
        """
        return iter(())


def test_similarity_retriever_returns_search_results():
    """
    Verify retrieve() returns SearchResult objects.
    """

    retriever = SimilarityRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    results = list(
        retriever.retrieve(
            query="Apache Spark",
        )
    )

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk.content == "Apache Spark"


def test_similarity_retriever_passes_top_k():
    """
    Verify top_k overrides the configured default.
    """

    vector_store = FakeVectorStore()

    retriever = SimilarityRetriever(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
    )

    list(
        retriever.retrieve(
            query="Apache Spark",
            top_k=10,
        )
    )

    assert vector_store.last_top_k == 10


def test_similarity_retriever_uses_configured_top_k():
    """
    Verify RetrievalConfig provides the default top_k.
    """

    vector_store = FakeVectorStore()

    retriever = SimilarityRetriever(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        config=RetrievalConfig(
            top_k=7,
        ),
    )

    list(
        retriever.retrieve(
            query="Apache Spark",
        )
    )

    assert vector_store.last_top_k == 7


def test_similarity_retriever_passes_query_embedding():
    """
    Verify the generated QueryEmbedding is forwarded
    to the VectorStore.
    """

    vector_store = FakeVectorStore()

    retriever = SimilarityRetriever(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
    )

    list(
        retriever.retrieve(
            query="Apache Spark",
        )
    )

    assert vector_store.last_query_embedding is not None
    assert vector_store.last_query_embedding.model == "unit-test-model"
    assert vector_store.last_query_embedding.vector == [0.1, 0.2, 0.3]

def test_similarity_retriever_passes_filters():
    """
    Verify metadata filters are forwarded to the VectorStore.
    """

    vector_store = FakeVectorStore()

    retriever = SimilarityRetriever(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
    )

    filters = {
        "author": "Yogesh",
    }

    list(
        retriever.retrieve(
            query="Apache Spark",
            filters=filters,
        )
    )

    assert vector_store.last_filters == filters

def clear(self) -> None:
    """
    Clear the fake vector store.
    """
    self.last_top_k = None
    self.last_query_embedding = None
    self.last_filters = None

   