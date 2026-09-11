"""
Purpose
-------
Builds the default prompt for Retrieval-Augmented Generation (RAG).

Responsibilities
----------------
- Combine retrieved context into a single prompt.
- Append the user query.
- Produce a deterministic prompt for the LLM.

Does NOT
--------
- Retrieve documents.
- Generate embeddings.
- Call an LLM.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.models.search_result import SearchResult
from ragkit.prompts.prompt_builder import PromptBuilder

'''
=> This class implements build method of PromptBuilder interface. It just do
string join of search results
'''
class DefaultPromptBuilder(PromptBuilder):
    """
    Default implementation of PromptBuilder.
    """

    def build(
            self,
            query: str,
            search_results: Iterable[SearchResult],
    ) -> str:
        """
        Build the final prompt using retrieved context.
        """

        context_parts: list[str] = []

        for result in search_results:
            filename = result.chunk.metadata.get(
                "filename",
                "Unknown document",
            )

            context_parts.append(
                f"""SOURCE DOCUMENT: {filename}
                {result.chunk.content}"""
            )

        context = "\n\n".join(context_parts)

        return f"""You are a precise document question-answering assistant.
        
            Answer the user's question using ONLY the provided context.
        
            Rules:
            1. Do not use information that is not present in the context.
            2. Do not invent, assume, estimate, or guess missing facts.
            3. If the document explicitly states an answer, prefer that statement
               over calculating an answer from separate pieces of information.
            4. If the document gives both an explicit answer and supporting details,
               report the explicit answer first and use the details only as supporting
               evidence.
            5. Do not calculate dates or durations using an assumed "current date"
               unless the context explicitly provides the reference date.
            6. If a calculation is required, use only dates, numbers, and facts
               explicitly present in the context.
            7. If the context does not contain enough information to answer,
               say that the answer cannot be determined from the provided context.
            8. Do not mention or use information from other documents.
            9. Document filenames are valid source metadata.
            10. If a person's name appears in the question but not in the document
                text, you may use the source document filename to identify which
                document the question refers to.
            11. Do not assume that a person is associated with a document unless
                the filename or other source metadata identifies them.
            12. Always answer in at least one complete, well-formed sentence.
                Never reply with a single word, a bare number, or a sentence
                fragment - state the specific fact directly, then briefly
                support it with the relevant detail from the context.

            Context
            -------
            {context}
        
            Query
            -----
            {query}
            """