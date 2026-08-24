"""
Purpose
-------
Represents the result of a RAG query.

Responsibilities
----------------
- Store the original user query.
- Store the normalized query used for retrieval.
- Store the generated answer.
- Store the source chunks used to generate the answer.

Does NOT
--------
- Perform query normalization.
- Perform retrieval.
- Generate the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ragkit.models.search_result import SearchResult


@dataclass(frozen=True, slots=True)
class RAGSource:
    """
    Source information returned with a RAG answer.
    """

    document_id: UUID
    filename: str
    chunk_id: UUID
    score: float


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """
    Complete response returned by the RAG service.
    """

    answer: str
    original_query: str
    normalized_query: str
    sources: list[RAGSource]