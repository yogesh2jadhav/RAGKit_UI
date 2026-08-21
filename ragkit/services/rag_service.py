"""
Purpose
-------
Provides a reusable application service for RAG queries.

Responsibilities
----------------
- Execute semantic vector retrieval.
- Execute BM25 keyword retrieval.
- Combine results using Reciprocal Rank Fusion.
- Build the LLM prompt.
- Generate the final LLM response.

Does NOT
--------
- Load documents.
- Create embeddings for documents.
- Store documents.
- Handle CLI input/output.
- Handle HTTP requests.
"""

from __future__ import annotations

from typing import Any

from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.llms.llm import LLM
from ragkit.models.llm_response import LLMResponse
from ragkit.prompts.prompt_builder import PromptBuilder
from ragkit.ranking.reciprocal_rank_fusion import ReciprocalRankFusion
from ragkit.retrievers.retriever import Retriever


class RAGService:
    """
    Application service for executing RAG queries.

    The service hides the RAG orchestration from callers.

    A CLI application, Web API, or another application can
    call this service without knowing how retrieval works.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        keyword_searcher: BM25Searcher,
        rrf: ReciprocalRankFusion,
        prompt_builder: PromptBuilder,
        llm: LLM,
        top_k: int = 5,
    ) -> None:
        """
        Initialize the RAG service.

        Parameters
        ----------
        retriever
            Semantic/vector retriever.

        keyword_searcher
            BM25 keyword searcher.

        rrf
            Reciprocal Rank Fusion implementation.

        prompt_builder
            Builds the final LLM prompt.

        llm
            Generates the final answer.

        top_k
            Maximum number of results returned by each
            retrieval strategy and by RRF.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self._retriever = retriever
        self._keyword_searcher = keyword_searcher
        self._rrf = rrf
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._top_k = top_k

    def ask(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Execute a RAG query.

        Flow
        ----
        Query
          ↓
        Vector Search
          +
        BM25
          ↓
        RRF
          ↓
        Prompt Builder
          ↓
        LLM
          ↓
        LLMResponse

        Parameters
        ----------
        query
            User's question.

        filters
            Optional metadata filters.

            These will be used by the semantic retriever.
            BM25 filtering will be handled later when we
            implement document selection.

        Returns
        -------
        LLMResponse
            Final generated response.
        """

        if not query.strip():
            raise ValueError("query must not be empty.")

        #
        # --------------------------------------------------------
        # 1. VECTOR SEARCH
        # --------------------------------------------------------
        #

        vector_results = list(
            self._retriever.retrieve(
                query=query,
                top_k=self._top_k,
                filters=filters,
            )
        )

        #
        # --------------------------------------------------------
        # 2. BM25 SEARCH
        # --------------------------------------------------------
        #

        keyword_results = list(
            self._keyword_searcher.search(
                query=query,
                top_k=self._top_k,
            )
        )

        #
        # --------------------------------------------------------
        # 3. RRF FUSION
        # --------------------------------------------------------
        #

        rrf_results = self._rrf.fuse(
            [
                vector_results,
                keyword_results,
            ],
            top_k=self._top_k,
        )

        #
        # --------------------------------------------------------
        # 4. BUILD PROMPT
        # --------------------------------------------------------
        #

        prompt = self._prompt_builder.build(
            query=query,
            search_results=rrf_results,
        )

        #
        # --------------------------------------------------------
        # 5. GENERATE ANSWER
        # --------------------------------------------------------
        #

        return self._llm.generate(
            prompt=prompt,
        )