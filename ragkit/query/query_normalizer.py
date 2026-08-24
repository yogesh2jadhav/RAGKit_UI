"""
Purpose
-------
Defines the interface for normalizing user queries.

Responsibilities
----------------
- Accept a user's original query.
- Return an original and normalized query.

Does NOT
--------
- Perform document retrieval.
- Perform BM25 search.
- Generate the final RAG answer.
- Decide which documents should be searched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.models.normalized_query import NormalizedQuery


class QueryNormalizer(ABC):
    """
    Interface for query normalization.
    """

    @abstractmethod
    def normalize(
        self,
        query: str,
    ) -> NormalizedQuery:
        """
        Normalize a user query.
        """
        raise NotImplementedError