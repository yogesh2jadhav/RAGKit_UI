from ragkit.models.normalized_query import NormalizedQuery


def test_normalized_query_stores_original_and_normalized_values():
    """
    Verify both original and normalized queries are preserved.
    """

    result = NormalizedQuery(
        original="How many years of experiance Yogesh have?",
        normalized="How many years of experience does Yogesh have?",
    )

    assert (
        result.original
        == "How many years of experiance Yogesh have?"
    )

    assert (
        result.normalized
        == "How many years of experience does Yogesh have?"
    )