"""
Purpose
-------
HTTP endpoint for RAG chat.

Responsibilities
----------------
- Validate chat requests.
- Pass the question and document selection to RAGService.
- Convert RAGResponse into an HTTP response.

Does NOT
--------
- Perform retrieval.
- Perform RRF.
- Build prompts.
- Call the LLM directly.
"""

from __future__ import annotations

from fastapi import APIRouter

from ragkit.api.models import (
    ChatRequest,
    ChatResponse,
    ChatSource,
)
from ragkit.services.rag_service import RAGService


def create_chat_router(
    rag_service: RAGService,
) -> APIRouter:
    """
    Create the chat API router.
    """

    router = APIRouter(
        prefix="/api/chat",
        tags=["chat"],
    )

    @router.post(
        "",
        response_model=ChatResponse,
    )
    def chat(
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Execute a RAG query.
        """

        response = rag_service.ask(
            request.question,
            document_ids=request.document_ids,
        )

        return ChatResponse(
            answer=response.answer,

            original_query=response.original_query,

            normalized_query=response.normalized_query,

            sources=[
                ChatSource(
                    document_id=source.document_id,
                    filename=source.filename,
                    chunk_id=source.chunk_id,
                    score=source.score,
                )
                for source in response.sources
            ],
        )

    return router