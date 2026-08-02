"""
Purpose
-------
Hybrid retriever combining semantic and keyword search.

Responsibilities
----------------
- Retrieve documents using an existing Retriever.
- Perform keyword search.
- Merge both result sets.
- Remove duplicate chunks.

Does NOT
--------
- Generate embeddings.
- Perform vector search directly.
- Build prompts.
- Call an LLM.
"""

from __future__ import annotations
from ragkit.ranking.rank_fusion import RankFusion
from ragkit.ranking.reciprocal_rank_fusion import ReciprocalRankFusion

from typing import Any

from ragkit.keyword.keyword_searcher import KeywordSearcher
from ragkit.models.search_result import SearchResult
from ragkit.retrievers.retriever import Retriever


class HybridRetriever(Retriever):
    """
    Hybrid semantic + keyword retriever.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        keyword_searcher: KeywordSearcher,
        rank_fusion: RankFusion | None = None,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        retriever
            Retriever used for semantic retrieval.

        keyword_searcher
            Keyword search implementation.
        """

        self._semantic_retriever = retriever
        self._keyword_searcher = keyword_searcher

        if rank_fusion is None:
            rank_fusion = ReciprocalRankFusion()

        self._rank_fusion = rank_fusion

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve documents using hybrid retrieval.
        """

        vector_results = list(
            self._semantic_retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
            )
        )

        keyword_results = list(
            self._keyword_searcher.search(
                query=query,
                top_k=top_k,
            )
        )

        return self._rank_fusion.fuse(
            [
                vector_results,
                keyword_results,
            ],
            top_k=top_k,
        )