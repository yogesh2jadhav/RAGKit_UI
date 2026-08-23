"""
Purpose
-------
Indexes documents from a Source into a VectorStore.

Responsibilities
----------------
- Discover source documents.
- Load documents.
- Chunk documents.
- Generate embeddings.
- Store embeddings in the VectorStore.
- Return indexing statistics.

Does NOT
--------
- Perform similarity search.
- Generate LLM responses.
"""

from __future__ import annotations

from ragkit.indexers.indexer import Indexer
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.models.indexing_result import IndexingResult
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.sources.source import Source
from ragkit.vectorstores.vector_store import VectorStore


class DocumentIndexer(Indexer):
    """
    Default implementation of Indexer.
    => Thie class shows load directory -> Read document -> create chunk ->
    do embeddings -> save to vector store.
    """

    def __init__(
        self,
        *,
        processor: DocumentProcessor,
        vector_store: VectorStore,
    ) -> None:
        """
        Parameters
        ----------
        processor
            Processes loaded documents into chunks
            and embeddings.

        vector_store
            Stores generated embeddings.
        """

        self._processor = processor
        self._vector_store = vector_store

    def index(
        self,
        source: Source,
    ) -> IndexingResult:
        """
        Index every document discovered from the supplied source.
        """

        document_count = 0
        chunk_count = 0
        embedding_count = 0

        #
        # Process one document at a time.
        #
        # This keeps memory usage nearly constant,
        # regardless of the total number of documents.
        #
        for source_document in source.discover():

            loader = LoaderFactory.get_loader(
                source_document,
            )

            document = loader.load(
                source_document,
            )
            document_count += 1
            chunks, embeddings = self._processor.process(
                document,
            )

            chunk_count += len(chunks)
            embedding_count += len(embeddings)

            self._vector_store.add(
                chunks=chunks,
                embeddings=embeddings,
            )

        return IndexingResult(
            documents=document_count,
            chunks=chunk_count,
            embeddings=embedding_count,
        )
