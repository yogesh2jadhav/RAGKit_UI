"""
Purpose
-------
HTTP request and response models for the RAGKit API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from uuid import UUID


class ChatRequest(BaseModel):
    """
    Request body for a RAG chat query.
    """

    question: str = Field(
        min_length=1,
    )

    document_ids: list[UUID] | None = None


class ChatSource(BaseModel):
    """
    Source information returned to the client.
    """

    document_id: UUID
    filename: str
    chunk_id: UUID
    score: float


class ChatResponse(BaseModel):
    """
    Response returned by the RAG chat endpoint.
    """

    answer: str
    sources: list[ChatSource]