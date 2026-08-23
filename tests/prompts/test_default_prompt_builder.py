from uuid import uuid4

from ragkit.models.chunk import Chunk
from ragkit.models.search_result import SearchResult
from ragkit.prompts.default_prompt_builder import DefaultPromptBuilder


def create_result(content: str) -> SearchResult:
    """
    Create a SearchResult for testing.
    """

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={},
    )

    return SearchResult(
        chunk=chunk,
        score=0.1,
    )


def test_build_prompt():
    """
    Verify a prompt is created from retrieved chunks.
    """

    builder = DefaultPromptBuilder()

    prompt = builder.build(
        query="What is Apache Spark?",
        search_results=[
            create_result(
                "Apache Spark is a distributed computing engine."
            ),
            create_result(
                "Spark supports batch and streaming workloads."
            ),
        ],
    )

    assert "Apache Spark is a distributed computing engine." in prompt

    assert "Spark supports batch and streaming workloads." in prompt

    assert "What is Apache Spark?" in prompt

    assert "Context" in prompt

    assert "Query" in prompt


def test_build_prompt_with_no_results():
    """
    Verify a prompt is still created when no
    search results are available.
    """

    builder = DefaultPromptBuilder()

    prompt = builder.build(
        query="Unknown question",
        search_results=[],
    )

    assert "Unknown question" in prompt

    assert "Context" in prompt

    assert "Query" in prompt

def test_prompt_prioritizes_explicit_document_answer():
    """
    Verify the prompt instructs the LLM to prefer an
    explicitly stated answer over an unsupported calculation.
    """

    prompt = DefaultPromptBuilder().build(
        query="How many years of experience does Yogesh have?",
        search_results=[],
    )

    assert (
        "If the document explicitly states an answer, prefer that statement"
        in prompt
    )

    assert (
        "Do not invent, assume, estimate, or guess missing facts."
        in prompt
    )

    assert (
        "Do not calculate dates or durations using an assumed"
        in prompt
    )