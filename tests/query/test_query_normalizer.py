from ragkit.models.normalized_query import NormalizedQuery
from ragkit.query.query_normalizer import QueryNormalizer


class FakeQueryNormalizer(QueryNormalizer):
    """
    Fake normalizer used to verify the interface.
    """

    def normalize(
        self,
        query: str,
    ) -> NormalizedQuery:
        return NormalizedQuery(
            original=query,
            normalized="normalized query",
        )


def test_query_normalizer_returns_normalized_query():
    """
    Verify a QueryNormalizer returns NormalizedQuery.
    """

    normalizer = FakeQueryNormalizer()

    result = normalizer.normalize(
        "original query",
    )

    assert result.original == "original query"
    assert result.normalized == "normalized query"