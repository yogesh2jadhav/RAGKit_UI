"""
Purpose
-------
Defines the abstract interface for all vector store implementations.

Responsibilities
----------------
- Persist chunk embeddings.
- Retrieve similar chunks for a query.

Does NOT
---------
- Generate embeddings.
- Perform chunking.
- Call an LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.search_result import SearchResult

'''
=> VectorStore is interface with 3 empty methods
    1. add
    2. search
    3. count
'''
class VectorStore(ABC):
    """
    Abstract base class for all vector stores.

    Every implementation is responsible for:

    - Persisting embeddings.
    - Retrieving similar chunks.

    The interface intentionally exposes Iterables instead
    of concrete collections. This keeps the API compatible
    with both eager (list-based) and streaming
    implementations.
    """

    @abstractmethod
    def add(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Embedding],
    ) -> None:
        """
        Persist chunks together with their embeddings.

        Parameters
        ----------
        chunks
            Original chunks produced by the Chunker.

        embeddings
            Embeddings corresponding to the supplied chunks.

        Notes
        -----
        Every Chunk must have exactly one matching Embedding.
        Matching is performed using Chunk.id and
        Embedding.chunk_id.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> Iterable[SearchResult]:
        """
        Retrieve the most relevant chunks.

        Parameters
        ----------
        query_embedding
            Embedding generated from the user's question.

        top_k
            Maximum number of results to retrieve.

        filters
            Optional metadata filters used to limit
            the search results.
        Returns
        -------
        Iterable[SearchResult]

            Search results ordered from best match to
            worst match.

            Returning an Iterable instead of a list allows
            future vector stores to stream results rather
            than materialising them all in memory.
        """
        raise NotImplementedError

    @abstractmethod
    def iter_chunks(
        self,
    ) -> Iterable[Chunk]:
        """
        Iterate over every indexed chunk.

        Returns
        -------
        Iterable[Chunk]

            All chunks currently stored in the vector store.

        Notes
        -----
        Returning an Iterable keeps the interface compatible
        with vector stores that stream results instead of
        loading the entire index into memory.
        """
        raise NotImplementedError


    @abstractmethod
    def count(self) -> int:
        """
        Return the number of stored vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all indexed data from the vector store.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_document(
        self,
        document_id: UUID,
    ) -> None:
        """
        Remove all indexed chunks belonging to a document.

        Parameters
        ----------
        document_id
            ID of the document whose chunks should be removed.
        """
        raise NotImplementedError