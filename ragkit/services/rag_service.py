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
from ragkit.models.rag_response import (
    RAGResponse,
    RAGSource,
)
from typing import Any

from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.llms.llm import LLM
from ragkit.logger import logger
from ragkit.models.llm_response import LLMResponse
from ragkit.prompts.prompt_builder import PromptBuilder
from ragkit.ranking.reciprocal_rank_fusion import ReciprocalRankFusion
from ragkit.retrievers.retriever import Retriever
from collections.abc import Iterable
from uuid import UUID
from ragkit.models.normalized_query import NormalizedQuery
from ragkit.query.query_normalizer import QueryNormalizer

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
            query_normalizer: QueryNormalizer | None = None,
            top_k: int = 5,
            candidate_k: int = 15,
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

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        self._retriever = retriever
        self._keyword_searcher = keyword_searcher
        self._rrf = rrf
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._query_normalizer = query_normalizer

    def ask(
            self,
            query: str,
            *,
            filters: dict[str, object] | None = None,
            document_ids: list[UUID] | None = None,
    ) -> RAGResponse:

        if not query.strip():
            raise ValueError(
                "query must not be empty."
            )

        #
        # --------------------------------------------------------
        # 0. NORMALIZE QUERY
        # --------------------------------------------------------
        #

        if self._query_normalizer is not None:
            normalized_query = (
                self._query_normalizer.normalize(
                    query,
                )
            )
        else:
            normalized_query = NormalizedQuery(
                original=query,
                normalized=query,
            )

        search_query = normalized_query.normalized

        logger.info(
            "RAG query: original=%r, normalized=%r",
            normalized_query.original,
            search_query,
        )



        #
        # --------------------------------------------------------
        # 1. VECTOR SEARCH
        # --------------------------------------------------------
        #

        vector_filters = dict(filters or {})

        if document_ids is not None:
            vector_filters["_ragkit_document_id"] = {
                "$in": [
                    str(document_id)
                    for document_id in document_ids
                ]
            }

        vector_results = list(
            self._retriever.retrieve(
                query=search_query,
                top_k=self._candidate_k,
                filters=vector_filters or None,
            )
        )

        #
        # --------------------------------------------------------
        # 2. BM25 SEARCH
        # --------------------------------------------------------
        #

        keyword_results = list(
            self._keyword_searcher.search(
                query=search_query,
                top_k=self._candidate_k,
                document_ids=document_ids,
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
            top_k=self._candidate_k,
        )

        final_results = rrf_results[:self._top_k]

        logger.info(
            "Retrieval: %d vector, %d keyword, %d after RRF (top_k=%d)",
            len(vector_results),
            len(keyword_results),
            len(final_results),
            self._top_k,
        )

        #
        # --------------------------------------------------------
        # 4. BUILD PROMPT
        # --------------------------------------------------------
        #

        prompt = prompt = self._prompt_builder.build(
            query=search_query,
            search_results=final_results,
        )

        #
        # --------------------------------------------------------
        # 5. GENERATE ANSWER
        # --------------------------------------------------------
        #

        logger.info("Generating answer with LLM")

        llm_response = self._llm.generate(
            prompt=prompt,
        )

        logger.info(
            "LLM answer generated (%d chars)",
            len(llm_response.content),
        )

        #
        # --------------------------------------------------------
        # 6. BUILD SOURCES
        # --------------------------------------------------------
        #

        sources = [
            RAGSource(
                document_id=result.chunk.document_id,
                filename=result.chunk.metadata.get(
                    "filename",
                    result.chunk.metadata.get(
                        "uri",
                        "unknown",
                    ),
                ),
                chunk_id=result.chunk.id,
                score=result.score,
            )
            for result in final_results
        ]

        return RAGResponse(
            answer=llm_response.content,
            original_query=normalized_query.original,
            normalized_query=normalized_query.normalized,
            sources=sources,
        )
