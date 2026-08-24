from ragkit.llms.llm import LLM
from ragkit.models.llm_response import LLMResponse
from ragkit.query.ollama_query_normalizer import (
    OllamaQueryNormalizer,
)


class FakeLLM(LLM):
    """
    Fake LLM used to test query normalization.
    """

    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def generate(
        self,
        prompt: str,
        options=None,
    ) -> LLMResponse:
        self.last_prompt = prompt

        return LLMResponse(
            content=self.response,
        )


def test_normalizer_corrects_spelling():
    """
    Verify an obvious spelling mistake is normalized.
    """

    llm = FakeLLM(
        "How many years of experience does Yogesh have?"
    )

    normalizer = OllamaQueryNormalizer(
        llm=llm,
    )

    result = normalizer.normalize(
        "How many years of experiance Yogesh have?"
    )

    assert (
        result.original
        == "How many years of experiance Yogesh have?"
    )

    assert (
        result.normalized
        == "How many years of experience does Yogesh have?"
    )


def test_normalizer_preserves_original_query():
    """
    Verify the original query is never modified.
    """

    query = (
        "How many years of experiance Yogesh have?"
    )

    llm = FakeLLM(
        "How many years of experience does Yogesh have?"
    )

    normalizer = OllamaQueryNormalizer(
        llm=llm,
    )

    result = normalizer.normalize(
        query,
    )

    assert result.original == query


def test_normalizer_preserves_technical_terms():
    """
    Verify the normalization prompt explicitly protects
    technical terminology.
    """

    llm = FakeLLM(
        "How much experience does Yogesh have with Databricks?"
    )

    normalizer = OllamaQueryNormalizer(
        llm=llm,
    )

    normalizer.normalize(
        "How much experiance does Yogesh have with Databricks?"
    )

    assert llm.last_prompt is not None

    assert "technical terms" in (
        llm.last_prompt
    )

    assert "Databricks" in (
        llm.last_prompt
    )


def test_normalizer_rejects_empty_query():
    """
    Verify empty queries are rejected.
    """

    llm = FakeLLM(
        "unused",
    )

    normalizer = OllamaQueryNormalizer(
        llm=llm,
    )

    try:
        normalizer.normalize("   ")
    except ValueError as exc:
        assert str(exc) == "query must not be empty."
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_normalizer_falls_back_to_original_when_llm_returns_empty():
    """
    Verify an empty LLM response does not produce an empty
    retrieval query.
    """

    query = (
        "How many years of experiance does Yogesh have?"
    )

    llm = FakeLLM("")

    normalizer = OllamaQueryNormalizer(
        llm=llm,
    )

    result = normalizer.normalize(
        query,
    )

    assert result.original == query
    assert result.normalized == query