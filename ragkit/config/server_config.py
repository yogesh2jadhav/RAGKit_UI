"""
Purpose
-------
Configuration for the RAGKit server application.

Responsibilities
----------------
- Define server-side VectorStore configuration.
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


def default_server_config() -> ServerConfig:
    """
    Return the default local development configuration.

    The default database points to the existing RRF example
    database so the API can immediately see the documents
    already indexed during development.
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
    )