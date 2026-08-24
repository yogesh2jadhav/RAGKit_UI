"""
Purpose
-------
Normalizes user queries using an LLM.

Responsibilities
----------------
- Send a normalization instruction to an LLM.
- Preserve the original query.
- Return the normalized query.

Does NOT
--------
- Retrieve documents.
- Perform BM25 search.
- Perform vector search.
- Generate the final RAG answer.
"""

from __future__ import annotations

from ragkit.llms.llm import LLM
from ragkit.models.normalized_query import NormalizedQuery
from ragkit.query.query_normalizer import QueryNormalizer


class OllamaQueryNormalizer(QueryNormalizer):
    """
    QueryNormalizer implementation that uses an LLM.

    The implementation depends on the generic LLM interface,
    so it is not coupled directly to Ollama.
    """

    def __init__(
        self,
        *,
        llm: LLM,
    ) -> None:
        """
        Initialize the query normalizer.
        """

        self._llm = llm

    def normalize(
        self,
        query: str,
    ) -> NormalizedQuery:
        """
        Normalize a user's query.

        The LLM is instructed to:
        - Correct obvious spelling mistakes.
        - Correct basic grammar.
        - Preserve names.
        - Preserve technical terms.
        - Preserve company/product names.
        - Preserve acronyms.
        - Preserve the original meaning.
        - Return only the corrected query.
        """

        if not query.strip():
            raise ValueError(
                "query must not be empty."
            )

        prompt = self._build_prompt(
            query,
        )

        response = self._llm.generate(
            prompt=prompt,
        )

        normalized = response.content.strip()

        if not normalized:
            normalized = query.strip()

        return NormalizedQuery(
            original=query,
            normalized=normalized,
        )

    @staticmethod
    def _build_prompt(
        query: str,
    ) -> str:
        """
        Build the query-normalization prompt.
        """

        return f"""
            You are a query normalization component for a RAG system.
            
            Normalize the user's question before document retrieval.
            
            Rules:
            1. Correct obvious spelling mistakes.
            2. Correct obvious grammar mistakes.
            3. Preserve the original meaning.
            4. Preserve people's names exactly.
            5. Preserve company names exactly.
            6. Preserve product names exactly.
            7. Preserve technical terms exactly when they appear intentional.
            8. Preserve acronyms such as RAG, BM25, LLM, API, SQL, JVM, and GPU.
            9. Do not answer the question.
            10. Do not explain your changes.
            11. Return ONLY the normalized question.
            
            User query:
            {query}
            """.strip()