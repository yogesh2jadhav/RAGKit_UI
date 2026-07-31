"""
Defines the abstract base class for all rank fusion algorithms.

A RankFusion implementation combines multiple ranked search result lists
into a single ranked list.

Examples
--------
Reciprocal Rank Fusion (RRF)
Weighted Reciprocal Rank Fusion
Borda Count
"""

from abc import ABC, abstractmethod

from ragkit.models.search_result import SearchResult


class RankFusion(ABC):
    """
    Abstract base class for rank fusion algorithms.

    A rank fusion algorithm combines the ranked outputs from multiple
    retrievers into a single ranked list.

    Notes
    -----
    A RankFusion implementation:

    - does NOT retrieve documents
    - does NOT compute embeddings
    - does NOT query a vector database

    It only merges already-ranked search results.
    """

    @abstractmethod
    def fuse(
        self,
        ranked_lists: list[list[SearchResult]], # List of multiple Retriever like SimilarityRetriever, BM25 or SQLRetriever.
            # Each retriver give Result in List that's why List of List
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Merge multiple ranked search result lists.

        Parameters
        ----------
        ranked_lists
            Ranked search results produced by one or more retrievers.

            Example::

                [
                    semantic_results,
                    keyword_results,
                    graph_results,
                ]

        top_k
            Maximum number of fused results to return.

            If None, return every fused result.

        Returns
        -------
        list[SearchResult]
            The fused ranking.
        """