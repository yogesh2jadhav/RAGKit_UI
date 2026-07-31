"""
Implementation of Reciprocal Rank Fusion (RRF).

References
----------
Cormack, Clarke, Buettcher

'Reciprocal Rank Fusion Outperforms Condorcet and
Individual Rank Learning Methods'

SIGIR 2009
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ragkit.models.search_result import SearchResult
from ragkit.ranking.rank_fusion import RankFusion


@dataclass(slots=True)
class _RankedResult:
    """
    Internal helper used while computing RRF scores.

    Attributes
    ----------
    result
        The original search result.

    score
        Accumulated Reciprocal Rank Fusion score.
    """

    result: SearchResult
    score: float = 0.0


class ReciprocalRankFusion(RankFusion):
    """
    Reciprocal Rank Fusion (RRF).

    RRF combines multiple ranked search result lists into
    a single ranking.

    Unlike score-based fusion, RRF ignores the retrieval
    scores produced by individual retrievers.

    Instead, it only considers the rank position of each
    search result.

    Score Formula
    -------------

        score += 1 / (k + rank)

    where

    - rank starts from 1
    - k defaults to 60
    """

    def __init__(
        self,
        k: int = 60,
    ) -> None:
        """
        Initialize the RRF algorithm.

        Parameters
        ----------
        k
            Rank constant used by the RRF formula.

            Larger values reduce the influence of
            high-ranked search results.

            The original paper recommends 60.
        """

        self._k = k

    def fuse(
        self,
        ranked_lists: list[list[SearchResult]],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Fuse multiple ranked search result lists.

        Parameters
        ----------
        ranked_lists
            Ranked search results returned by one or more
            retrievers.

        top_k
            Maximum number of fused search results to
            return.

            If None, every fused result is returned.

        Returns
        -------
        list[SearchResult]
            Search results sorted by descending RRF score.
        """

        #
        # Maps Chunk ID -> accumulated RRF score and
        # original SearchResult.
        #
        rankings: dict[UUID, _RankedResult] = {}

        #
        # Process each ranked list independently.
        #
        for ranked_list in ranked_lists:

            #
            # RRF ranks start from one.
            #
            for rank, result in enumerate(
                ranked_list,
                start=1,
            ):

                chunk_id = result.chunk.id

                #
                # Create the ranking entry only once.
                #
                entry = rankings.setdefault(
                    chunk_id,
                    _RankedResult(
                        result=result,
                    ),
                )

                #
                # Add this retriever's contribution.
                #
                entry.score += 1.0 / (
                    self._k + rank
                )

        #
        # Sort by descending RRF score.
        #
        ranked_results = sorted(
            rankings.values(),
            key=lambda item: item.score,
            reverse=True,
        )

        fused_results = [
            item.result
            for item in ranked_results
        ]

        if top_k is not None:
            fused_results = fused_results[:top_k]

        return fused_results