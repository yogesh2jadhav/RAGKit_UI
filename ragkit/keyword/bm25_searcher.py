"""
Purpose
-------
BM25 keyword search implementation.

Responsibilities
----------------
- Build a BM25 index.
- Perform lexical search.
- Return ranked search results.

Does NOT
--------
- Generate embeddings.
- Perform vector search.
- Call an LLM.
"""

from __future__ import annotations
from uuid import UUID
import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from ragkit.exceptions.keyword_searcher_error import KeywordSearcherError
from ragkit.keyword.keyword_searcher import KeywordSearcher
from ragkit.models.chunk import Chunk
from ragkit.models.search_result import SearchResult
from ragkit.vectorstores.vector_store import VectorStore


class BM25Searcher(KeywordSearcher):
    """
    BM25 keyword search implementation.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore,
    ) -> None:

        self._vector_store = vector_store

        self._chunks: list[Chunk] = []

        self._bm25: BM25Okapi | None = None

        self._build_index()

    def search(
            self,
            query: str,
            *,
            top_k: int = 5,
            document_ids: Iterable[UUID] | None = None,
    ) -> Iterable[SearchResult]:
        """
        Perform BM25 keyword search.

        When document_ids is provided, only chunks belonging
        to those documents are considered.
        """

        if self._bm25 is None:
            raise KeywordSearcherError(
                "BM25 index has not been built."
            )

        query_tokens = self._tokenize(
            query,
        )

        scores = self._bm25.get_scores(
            query_tokens,
        )

        selected_document_ids = (
            set(document_ids)
            if document_ids is not None
            else None
        )

        ranked = sorted(
            (
                (chunk, score)
                for chunk, score in zip(
                self._chunks,
                scores,
                strict=True,
            )
                if (
                    selected_document_ids is None
                    or chunk.document_id in selected_document_ids
            )
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        for chunk, score in ranked[:top_k]:
            yield SearchResult(
                chunk=chunk,
                score=float(score),
            )

    def _build_index(
        self,
    ) -> None:
        """
        Build the BM25 index.
        """

        chunks = list(self._vector_store.iter_chunks())

        self._chunks = chunks

        corpus = [
            self._tokenize(
                chunk.content,
            )
            for chunk in self._chunks
        ]

        self._bm25 = BM25Okapi(
            corpus,
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """
        Tokenize text.
        """

        return re.findall(
            r"\w+",
            text.lower(),
        )