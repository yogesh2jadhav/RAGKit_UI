"""
Purpose
-------
Configuration for the RAGKit server application.

Responsibilities
----------------
- Define server-side VectorStore configuration.
- Define server-side RAG model configuration.
- Keep API configuration separate from example applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """
    Configuration used by the RAGKit server.
    """

    vector_db_path: Path
    collection_name: str
    embedding_model: str
    llm_model: str
    retrieval_top_k: int


def default_server_config() -> ServerConfig:
    """
    Return the default local development configuration.
    """

    project_root = Path(__file__).resolve().parents[2]

    return ServerConfig(
        vector_db_path=(
            project_root
            / "examples"
            / "data"
            / "vector_db_rrf"
        ),
        collection_name="ragkit_rrf",

        # Use the same models configured by the
        # existing RRF CLI.
        embedding_model="nomic-embed-text",
        llm_model="qwen3:8b",
        retrieval_top_k=5,
    )