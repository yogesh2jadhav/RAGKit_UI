from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from ragkit.chunkers.chunker import Chunker
from ragkit.embeddings.embedder import Embedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.loaders.loader import Loader
from ragkit.models.chunk import Chunk
from ragkit.models.document import Document
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.source_document import SourceDocument
from ragkit.processors import DocumentProcessor
from ragkit.sources.source import Source
from ragkit.transformers.transformer import Transformer
from ragkit.vectorstores.vector_store import VectorStore


class FakeSource(Source):
    """
    Fake Source used for unit testing.
    """

    def discover(self) -> Iterable[SourceDocument]:

        yield SourceDocument(
            uri="docs/spark.txt",
            mime_type="text/plain",
        )


class FakeLoader(Loader):
    """
    Fake Loader used for unit testing.
    """

    @classmethod
    def supports(
        cls,
        source: SourceDocument,
    ) -> bool:
        return True

    def load(
        self,
        source: SourceDocument,
    ) -> Document:

        return Document(
            id=uuid4(),
            content="Apache Spark",
            metadata={},
        )


class FakeChunker(Chunker):
    """
    Fake Chunker used for unit testing.
    """

    def chunk(
        self,
        document: Document,
    ) -> Iterable[Chunk]:

        yield Chunk(
            id=uuid4(),
            document_id=document.id,
            index=0,
            content=document.content,
            start_offset=0,
            end_offset=len(document.content),
            metadata={},
        )


class FakeEmbedder(Embedder):
    """
    Fake Embedder used for unit testing.
    """

    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:

        for chunk in chunks:
            yield Embedding(
                chunk_id=chunk.id,
                model="unit-test",
                vector=[0.1, 0.2],
            )

    def embed_query(
        self,
        query: str,
    ) -> QueryEmbedding:

        return QueryEmbedding(
            model="unit-test",
            vector=[0.1, 0.2],
        )


class FakeVectorStore(VectorStore):
    """
    Fake VectorStore used for unit testing.
    """
    def delete_document(
        self,
        document_id: UUID,
    ) -> None:
        """
        Fake implementation required by VectorStore.
        """
        pass
    def __init__(self) -> None:

        self.add_called = False
        self.chunk_count = 0
        self.embedding_count = 0

    def add(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Embedding],
    ) -> None:

        chunk_list = list(chunks)
        embedding_list = list(embeddings)

        self.add_called = True
        self.chunk_count = len(chunk_list)
        self.embedding_count = len(embedding_list)

    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int = 5,
    ):
        return []

    def count(self) -> int:
        return self.embedding_count

    def iter_chunks(
        self,
    ) -> Iterable[Chunk]:
        return iter(())

    def clear(self) -> None:
        """
        Clear the fake vector store.
        """
        self.add_called = False
        self.chunk_count = 0
        self.embedding_count = 0


def test_document_indexer_indexes_documents(monkeypatch):
    """
    Verify DocumentIndexer orchestrates the
    complete indexing workflow.
    """

    from ragkit.loaders import loader_factory

    monkeypatch.setattr(
        loader_factory.LoaderFactory,
        "get_loader",
        lambda _: FakeLoader(),
    )

    vector_store = FakeVectorStore()

    processor = DocumentProcessor(
        transformer=FakeTransformer(),
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
    )

    indexer = DocumentIndexer(
        processor=processor,
        vector_store=vector_store,
    )

    result = indexer.index(
        FakeSource(),
    )

    assert result.documents == 1
    assert result.chunks == 1
    assert result.embeddings == 1

    assert vector_store.add_called is True
    assert vector_store.chunk_count == 1
    assert vector_store.embedding_count == 1

class FakeTransformer(Transformer):
    """
    Fake Transformer used for unit testing.
    """

    def __init__(self):

        self.called = False

    def transform(
        self,
        document: Document,
    ) -> Document:

        self.called = True

        return document


def test_document_indexer_uses_transformer(
    monkeypatch,
):
    """
    Verify the transformer is invoked before chunking.
    """

    from ragkit.loaders import loader_factory
    from ragkit.loaders.loader import Loader


    class FakeLoader(Loader):

        @classmethod
        def supports(
            cls,
            source,
        ) -> bool:
            return True

        def load(
            self,
            source,
        ) -> Document:

            return Document(
                id=uuid4(),
                content="Apache Spark",
                metadata={},
            )


    monkeypatch.setattr(
        loader_factory.LoaderFactory,
        "get_loader",
        lambda _: FakeLoader(),
    )

    transformer = FakeTransformer()

    processor = DocumentProcessor(
        transformer=transformer,
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
    )

    indexer = DocumentIndexer(
        processor=processor,
        vector_store=FakeVectorStore(),
    )

    indexer.index(
        FakeSource(),
    )

    assert transformer.called

def clear(self) -> None:
    """
    Clear the fake vector store.
    """
    self.add_called = False
    self.chunk_count = 0
    self.embedding_count = 0