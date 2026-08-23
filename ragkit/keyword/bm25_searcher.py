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

import re
from collections.abc import Iterable
from uuid import UUID

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

    def rebuild(self) -> None:
        """
        Rebuild the BM25 index from the current VectorStore.

        This is used after new documents are indexed so that
        BM25 can immediately search the newly added chunks.
        """

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
            return

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

        An empty VectorStore is valid. In that case,
        no BM25 index is created until documents are indexed.
        """

        chunks = list(
            self._vector_store.iter_chunks()
        )

        self._chunks = chunks

        if not chunks:
            self._bm25 = None
            return

        corpus = [
            self._tokenize(
                self._searchable_text(chunk),
            )
            for chunk in self._chunks
        ]

        self._bm25 = BM25Okapi(
            corpus,
        )

    @staticmethod
    def _searchable_text(
        chunk: Chunk,
    ) -> str:
        """
        Build the text indexed by BM25.

        Includes both document metadata and chunk content
        so document filenames can participate in lexical search.
        """

        filename = str(
            chunk.metadata.get(
                "filename",
                "",
            )
        )

        return f"{filename} {chunk.content}"

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