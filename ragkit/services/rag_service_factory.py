"""
Purpose
-------
Creates the production RAGService and its dependencies.

Responsibilities
----------------
- Construct the embedding model.
- Construct the semantic retriever.
- Construct the BM25 searcher.
- Construct RRF.
- Construct the prompt builder.
- Construct the LLM.
- Return a ready-to-use RAGService.

Does NOT
--------
- Handle HTTP requests.
- Handle CLI input/output.
- Execute queries.
"""

from __future__ import annotations

from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.llms.ollama_llm import OllamaLLM
from ragkit.prompts.default_prompt_builder import DefaultPromptBuilder
from ragkit.ranking.reciprocal_rank_fusion import ReciprocalRankFusion
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.services.rag_service import RAGService
from ragkit.vectorstores.vector_store import VectorStore


def create_rag_service(
    *,
    vector_store: VectorStore,
    embedding_model: str,
    llm_model: str,
    top_k: int = 5,
    bm25_searcher: BM25Searcher | None = None,
) -> RAGService:
    """
    Create a fully configured RAGService.

    Parameters
    ----------
    vector_store
        Vector store used for semantic retrieval.

    embedding_model
        Ollama embedding model name.

    llm_model
        Ollama LLM model name.

    top_k
        Number of final retrieval results.

    bm25_searcher
        Optional existing BM25Searcher.

        If supplied, the RAGService uses that instance.
        This allows other application services, such as
        DocumentService, to rebuild the same BM25 index
        after document uploads.
    """

    embedder = OllamaEmbedder(
        model=embedding_model,
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    if bm25_searcher is None:
        bm25_searcher = BM25Searcher(
            vector_store=vector_store,
        )

    rrf = ReciprocalRankFusion()

    prompt_builder = DefaultPromptBuilder()

    llm = OllamaLLM(
        model=llm_model,
    )

    return RAGService(
        retriever=retriever,
        keyword_searcher=bm25_searcher,
        rrf=rrf,
        prompt_builder=prompt_builder,
        llm=llm,
        top_k=top_k,
    )