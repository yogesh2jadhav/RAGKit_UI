from collections.abc import Iterable
from uuid import UUID, uuid4

import pytest

from ragkit.llms.llm import LLM
from ragkit.models.chunk import Chunk
from ragkit.models.llm_response import LLMResponse
from ragkit.models.rag_response import RAGResponse
from ragkit.models.search_result import SearchResult
from ragkit.prompts.prompt_builder import PromptBuilder
from ragkit.ranking.reciprocal_rank_fusion import ReciprocalRankFusion
from ragkit.retrievers.retriever import Retriever
from ragkit.services.rag_service import RAGService


class FakeRetriever(Retriever):
    """
    Fake semantic retriever used for testing.
    """

    def __init__(
        self,
        results: list[SearchResult] | None = None,
    ) -> None:
        self.results = results or []
        self.last_query = None
        self.last_top_k = None
        self.last_filters = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> Iterable[SearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        self.last_filters = filters

        return self.results[:top_k]


class FakeBM25Searcher:
    """
    Fake BM25 searcher used for testing.
    """

    def __init__(
        self,
        results: list[SearchResult] | None = None,
    ) -> None:
        self.results = results or []
        self.last_query = None
        self.last_top_k = None
        self.last_document_ids = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids=None,
    ) -> Iterable[SearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        self.last_document_ids = document_ids

        return self.results[:top_k]


class FakeRRF:
    """
    Fake Reciprocal Rank Fusion implementation used for testing.
    """

    def __init__(self) -> None:
        self.last_ranked_lists = None
        self.last_top_k = None

    def fuse(
        self,
        ranked_lists,
        top_k=None,
    ):
        self.last_ranked_lists = ranked_lists
        self.last_top_k = top_k

        results = []

        for ranked_list in ranked_lists:
            results.extend(ranked_list)

        if top_k is None:
            return results

        return results[:top_k]


class FakePromptBuilder:
    """
    Fake prompt builder used for testing.
    """

    def __init__(self) -> None:
        self.last_query = None
        self.last_search_results = []

    def build(
        self,
        *,
        query,
        search_results,
    ):
        self.last_query = query
        self.last_search_results = list(search_results)

        return "TEST PROMPT"


class FakeLLM(LLM):
    """
    Fake LLM used for testing.
    """

    def __init__(self) -> None:
        self.last_prompt = None

    def generate(
        self,
        *,
        prompt: str,
    ) -> LLMResponse:
        self.last_prompt = prompt

        return LLMResponse(
            content="TEST ANSWER",
        )


def create_search_result(
    *,
    index: int = 0,
    filename: str = "test.docx",
) -> SearchResult:
    """
    Create a deterministic SearchResult for tests.
    """

    document_id = uuid4()
    chunk_id = uuid4()

    chunk = Chunk(
        id=chunk_id,
        document_id=document_id,
        index=index,
        content=f"Test content {index}",
        start_offset=0,
        end_offset=len(f"Test content {index}"),
        metadata={
            "filename": filename,
        },
    )

    return SearchResult(
        chunk=chunk,
        score=float(index + 1),
    )


def create_service(
    *,
    top_k: int = 5,
    candidate_k: int = 15,
    retriever_results: list[SearchResult] | None = None,
    keyword_results: list[SearchResult] | None = None,
):
    """
    Create RAGService with fake dependencies.
    """

    if retriever_results is None:
        retriever_results = [
            create_search_result(
                index=index,
                filename="vector.docx",
            )
            for index in range(candidate_k)
        ]

    if keyword_results is None:
        keyword_results = [
            create_search_result(
                index=index,
                filename="keyword.docx",
            )
            for index in range(candidate_k)
        ]

    retriever = FakeRetriever(
        results=retriever_results,
    )

    keyword_searcher = FakeBM25Searcher(
        results=keyword_results,
    )

    rrf = FakeRRF()
    prompt_builder = FakePromptBuilder()
    llm = FakeLLM()

    service = RAGService(
        retriever=retriever,
        keyword_searcher=keyword_searcher,
        rrf=rrf,
        prompt_builder=prompt_builder,
        llm=llm,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    return (
        service,
        retriever,
        keyword_searcher,
        rrf,
        prompt_builder,
        llm,
    )


def test_rag_service_returns_llm_response():
    """
    Verify RAGService returns the response generated by the LLM.
    """

    (
        service,
        _,
        _,
        _,
        _,
        llm,
    ) = create_service()

    response = service.ask(
        "How many years of experience?",
    )

    assert isinstance(response, RAGResponse)
    assert response.answer == "TEST ANSWER"
    assert llm.last_prompt == "TEST PROMPT"


def test_rag_service_passes_query_to_retrievers():
    """
    Verify the user's query is passed to both retrieval strategies.
    """

    (
        service,
        retriever,
        keyword_searcher,
        _,
        _,
        _,
    ) = create_service()

    service.ask(
        "How many years of experience?",
    )

    assert retriever.last_query == (
        "How many years of experience?"
    )

    assert keyword_searcher.last_query == (
        "How many years of experience?"
    )


def test_rag_service_passes_candidate_k_to_retrieval_stages():
    """
    Verify candidate_k is passed to all retrieval stages.
    """

    (
        service,
        retriever,
        keyword_searcher,
        rrf,
        _,
        _,
    ) = create_service(
        top_k=10,
        candidate_k=20,
    )

    service.ask(
        "How many years of experience?",
    )

    assert retriever.last_top_k == 20
    assert keyword_searcher.last_top_k == 20
    assert rrf.last_top_k == 20


def test_rag_service_passes_filters_to_retriever():
    """
    Verify metadata filters are forwarded to semantic retrieval.
    """

    (
        service,
        retriever,
        _,
        _,
        _,
        _,
    ) = create_service()

    filters = {
        "author": "Yogesh",
    }

    service.ask(
        "How many years of experience?",
        filters=filters,
    )

    assert retriever.last_filters == filters


def test_rag_service_passes_results_to_rrf():
    """
    Verify vector and BM25 results are passed to RRF.
    """

    (
        service,
        _,
        _,
        rrf,
        _,
        _,
    ) = create_service()

    service.ask(
        "Apache Spark",
    )

    assert rrf.last_ranked_lists is not None
    assert len(rrf.last_ranked_lists) == 2


def test_rag_service_builds_prompt_from_rrf_results():
    """
    Verify the RRF results are passed to PromptBuilder.
    """

    (
        service,
        _,
        _,
        _,
        prompt_builder,
        _,
    ) = create_service()

    service.ask(
        "Apache Spark",
    )

    assert prompt_builder.last_query == "Apache Spark"
    assert prompt_builder.last_search_results


def test_rag_service_rejects_empty_query():
    """
    Verify empty queries are rejected.
    """

    (
        service,
        _,
        _,
        _,
        _,
        _,
    ) = create_service()

    with pytest.raises(ValueError):
        service.ask("")


def test_rag_service_rejects_whitespace_query():
    """
    Verify whitespace-only queries are rejected.
    """

    (
        service,
        _,
        _,
        _,
        _,
        _,
    ) = create_service()

    with pytest.raises(ValueError):
        service.ask("   ")


def test_rag_service_rejects_invalid_top_k():
    """
    Verify top_k must be greater than zero.
    """

    with pytest.raises(ValueError):
        create_service(
            top_k=0,
        )


def test_rag_service_rejects_invalid_candidate_k():
    """
    Verify candidate_k must be greater than zero.
    """

    with pytest.raises(ValueError):
        create_service(
            candidate_k=0,
        )


def test_rag_service_rejects_candidate_k_smaller_than_top_k():
    """
    Verify candidate_k cannot be smaller than top_k.
    """

    with pytest.raises(ValueError):
        create_service(
            top_k=10,
            candidate_k=5,
        )


def test_rag_service_passes_document_ids_to_bm25():
    """
    Verify selected document IDs are forwarded to BM25 search.
    """

    (
        service,
        _,
        keyword_searcher,
        _,
        _,
        _,
    ) = create_service()

    document_ids = [
        uuid4(),
        uuid4(),
    ]

    service.ask(
        "Apache Spark",
        document_ids=document_ids,
    )

    assert keyword_searcher.last_document_ids == document_ids


def test_rag_service_passes_document_ids_to_vector_retriever():
    """
    Verify selected document IDs are converted into
    a VectorStore metadata filter.
    """

    (
        service,
        retriever,
        _,
        _,
        _,
        _,
    ) = create_service()

    document_id = uuid4()

    service.ask(
        "How many years of experience?",
        document_ids=[document_id],
    )

    assert retriever.last_filters == {
        "_ragkit_document_id": {
            "$in": [
                str(document_id),
            ],
        },
    }


def test_rag_service_returns_sources():
    """
    Verify RAGService returns sources from RRF results.
    """

    document_id = uuid4()
    chunk_id = uuid4()

    chunk = Chunk(
        id=chunk_id,
        document_id=document_id,
        index=0,
        content="Yogesh has nearly 20 years of experience.",
        start_offset=0,
        end_offset=48,
        metadata={
            "filename": "Yogesh Ashok 007.docx",
        },
    )

    result = SearchResult(
        chunk=chunk,
        score=0.95,
    )

    (
        service,
        _,
        _,
        rrf,
        _,
        _,
    ) = create_service()

    rrf.fuse = lambda ranked_lists, top_k=None: [
        result,
    ]

    response = service.ask(
        "How many years of experience?",
    )

    assert len(response.sources) == 1

    source = response.sources[0]

    assert source.document_id == document_id
    assert source.filename == "Yogesh Ashok 007.docx"
    assert source.chunk_id == chunk_id
    assert source.score == 0.95


def test_rag_service_limits_final_results_to_top_k():
    """
    Verify top_k controls the final number of results
    passed to the prompt builder.
    """

    (
        service,
        _,
        _,
        _,
        prompt_builder,
        _,
    ) = create_service(
        top_k=5,
        candidate_k=15,
    )

    service.ask(
        "How many years of experience?",
    )

    assert len(prompt_builder.last_search_results) == 5