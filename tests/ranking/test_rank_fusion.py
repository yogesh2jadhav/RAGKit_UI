from pytest import raises

from ragkit.ranking.rank_fusion import RankFusion


class TestRankFusion:
    """Tests for RankFusion."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """RankFusion cannot be instantiated directly."""

        with raises(TypeError):
            RankFusion()