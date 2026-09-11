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

import os
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
    llm_think: bool = False


def _env_bool(name: str, default: bool) -> bool:
    """
    Read a boolean environment variable.

    Accepts "1"/"true"/"yes"/"on" (case-insensitive) as True and
    "0"/"false"/"no"/"off" as False. Falls back to ``default`` if unset
    or unrecognized.
    """

    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_server_config() -> ServerConfig:
    """
    Return the default local development configuration.
    """

    project_root = Path(__file__).resolve().parents[2]

    return ServerConfig(
        vector_db_path=(
            project_root
            / "documents"
            / "data"
            / "vector_db_rrf"
        ),
        collection_name="ragkit_rrf",

        # Use the same models configured by the
        # existing RRF CLI.
        embedding_model="nomic-embed-text",
        llm_model="qwen3:8b",
        retrieval_top_k=5,

        # Reasoning models (qwen3, deepseek-r1, ...) generate a long
        # "thinking" trace before the final answer, which is slow on
        # CPU-only machines but tends to produce more thorough answers.
        # Default to False for speed; set RAGKIT_LLM_THINK=true to
        # trade latency for answer quality.
        llm_think=_env_bool("RAGKIT_LLM_THINK", False),
    )