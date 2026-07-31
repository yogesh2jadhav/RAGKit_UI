from uuid import uuid4

from ragkit.models.chunk import Chunk
from ragkit.models.search_result import SearchResult
from ragkit.ranking.reciprocal_rank_fusion import ReciprocalRankFusion


class TestReciprocalRankFusion:
    """Tests for ReciprocalRankFusion."""

    @staticmethod
    def _create_result(content: str) -> SearchResult:
        """Create a SearchResult for testing."""

        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            index=0,
            content=content,
            start_offset=0,
            end_offset=len(content),
        )

        return SearchResult(
            chunk=chunk,
            score=1.0,
        )

    def test_empty_ranked_lists(self) -> None:
        """Fusing no ranked lists should return an empty list."""

        rrf = ReciprocalRankFusion()

        assert rrf.fuse([]) == []

    def test_single_ranked_list_preserves_order(self) -> None:
        """A single ranked list should preserve its ranking."""

        a = self._create_result("A")
        b = self._create_result("B")
        c = self._create_result("C")

        rrf = ReciprocalRankFusion()

        results = rrf.fuse([[a, b, c]])

        assert results == [a, b, c]

    def test_multiple_ranked_lists_without_overlap(self) -> None:
        """Documents from different retrievers should all appear."""

        a = self._create_result("A")
        b = self._create_result("B")
        c = self._create_result("C")
        d = self._create_result("D")

        rrf = ReciprocalRankFusion()

        results = rrf.fuse(
            [
                [a, b],
                [c, d],
            ]
        )

        assert len(results) == 4

        assert {r.chunk.id for r in results} == {
            a.chunk.id,
            b.chunk.id,
            c.chunk.id,
            d.chunk.id,
        }

    def test_duplicate_chunks_accumulate_scores(self) -> None:
        """
        Documents returned by multiple retrievers should receive
        a higher RRF score.
        """

        a = self._create_result("A")
        b = self._create_result("B")
        c = self._create_result("C")
        d = self._create_result("D")

        rrf = ReciprocalRankFusion()

        results = rrf.fuse(
            [
                [a, b, c],
                [c, d, a],
            ]
        )

        #
        # A appears twice:
        #
        #   rank 1 -> 1/61
        #   rank 3 -> 1/63
        #
        # C appears twice:
        #
        #   rank 3 -> 1/63
        #   rank 1 -> 1/61
        #
        # Both should rank ahead of B and D.
        #
        assert results[0].chunk.id in {
            a.chunk.id,
            c.chunk.id,
        }
        assert results[1].chunk.id in {
            a.chunk.id,
            c.chunk.id,
        }

        ids = {result.chunk.id for result in results}

        assert b.chunk.id in ids
        assert d.chunk.id in ids

    def test_top_k_limits_results(self) -> None:
        """Only the requested number of results should be returned."""

        a = self._create_result("A")
        b = self._create_result("B")
        c = self._create_result("C")

        rrf = ReciprocalRankFusion()

        results = rrf.fuse(
            [
                [a, b, c],
            ],
            top_k=2,
        )

        assert len(results) == 2

    def test_custom_k_value(self) -> None:
        """The constructor should allow a custom RRF constant."""

        a = self._create_result("A")
        b = self._create_result("B")

        rrf = ReciprocalRankFusion(k=10)

        results = rrf.fuse([[a, b]])

        assert results == [a, b]