"""
Purpose
-------
Defines the interface for keyword-based search.

Responsibilities
----------------
- Perform keyword search.
- Return ranked search results.

Does NOT
--------
- Generate embeddings.
- Perform vector search.
- Merge search results.
- Call an LLM.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from uuid import UUID

from ragkit.models.search_result import SearchResult


class KeywordSearcher(ABC):
    """
    Abstract base class for keyword search.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_ids: Iterable[UUID] | None = None,
    ) -> Iterable[SearchResult]:
        """
        Perform keyword search.

        Parameters
        ----------
        query
            User query.

        top_k
            Maximum number of results.

        document_ids
            Optional document IDs used to restrict the search.

            None means search all documents.

        Returns
        -------
        Iterable[SearchResult]
            Ranked search results.
        """
        raise NotImplementedError