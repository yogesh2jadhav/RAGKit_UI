"""
Implementation of Reciprocal Rank Fusion (RRF).

References
----------
Cormack, Clarke, Buettcher
'Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods'
SIGIR 2009
"""

from collections import defaultdict
from uuid import UUID

from ragkit.models.search_result import SearchResult
from ragkit.ranking.rank_fusion import RankFusion


class ReciprocalRankFusion(RankFusion):
    """
    Reciprocal Rank Fusion (RRF).

    RRF combines multiple ranked search result lists into a single ranking.

    Unlike score-based fusion, RRF ignores the retrieval scores and
    instead uses only the rank position of each document.

    The score assigned to a document is:

        score += 1 / (k + rank)

    where:
        rank starts from 1
        k defaults to 60
    """

    def __init__(self, k: int = 60) -> None:
        """
        Initialize the RRF algorithm.

        Parameters
        ----------
        k
            Rank constant used by the RRF formula.

            Larger values reduce the influence of high-ranked documents.

            Default is 60 as recommended in the original paper.
        """
        self._k = k

    def fuse(
        self,
        ranked_lists: list[list[SearchResult]],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Fuse multiple ranked result lists into a single ranking.
        """

        # Accumulated RRF score for every unique chunk.
        scores: dict[UUID, float] = defaultdict(float)

        # Keeps the original SearchResult so it can be returned later.
        results: dict[UUID, SearchResult] = {}

        #
        # Process every ranked list independently.
        #
        for ranked_list in ranked_lists:

            #
            # Rank starts from 1 because the RRF paper defines:
            #
            #     score = 1 / (k + rank)
            #
            for rank, result in enumerate(ranked_list, start=1):

                chunk_id = result.chunk.id

                #
                # Add this retriever's contribution.
                #
                scores[chunk_id] += 1.0 / (self._k + rank)

                #
                # Keep one SearchResult instance for this chunk.
                #
                results[chunk_id] = result

        #
        # Sort by descending RRF score.
        #
        ranked_chunk_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        fused_results = [
            results[chunk_id]
            for chunk_id in ranked_chunk_ids
        ]

        if top_k is not None:
            fused_results = fused_results[:top_k]

        return fused_results