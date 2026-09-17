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
    llm_think: bool = True
    chunk_size: int = 2500
    chunk_overlap: int = 300


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


def _env_int(name: str, default: int) -> int:
    """
    Read an integer environment variable, falling back to ``default``
    if unset or not a valid integer.
    """

    value = os.environ.get(name)

    if value is None:
        return default

    try:
        return int(value.strip())
    except ValueError:
        return default


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
        retrieval_top_k=_env_int("RAGKIT_TOP_K", 15),

        # Reasoning models (qwen3, deepseek-r1, ...) generate a long
        # "thinking" trace before the final answer, which is slow on
        # CPU-only machines but tends to produce more thorough answers.
        # Default to True for answer quality; set RAGKIT_LLM_THINK=false
        # to trade answer quality for latency.
        llm_think=_env_bool("RAGKIT_LLM_THINK", True),

        # Larger chunks give the LLM more context per retrieved source,
        # at the cost of retrieval precision. 300/50 was too small - it
        # produced terse, thin answers compared to bigger chunk sizes.
        # Only affects documents indexed AFTER this is applied; existing
        # documents must be re-uploaded to be re-chunked.
        chunk_size=_env_int("RAGKIT_CHUNK_SIZE", 2500),
        chunk_overlap=_env_int("RAGKIT_CHUNK_OVERLAP", 300),
    )