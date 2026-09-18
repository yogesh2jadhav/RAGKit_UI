"""
Purpose
-------
Configuration for the RAGKit server application.

Responsibilities
----------------
- Define server-side VectorStore configuration.
- Compose the same per-component configs (ChunkerConfig, EmbeddingConfig,
  LLMConfig) used by the library/example code, with server-specific
  defaults and environment-variable overrides.
- Keep API configuration separate from example applications.

Does NOT
--------
- Duplicate chunking/embedding/LLM settings as separate fields. Earlier
  revisions of this file re-declared chunk_size/chunk_overlap/
  embedding_model/llm_model/llm_think as flat fields here, independent
  of ChunkerConfig/EmbeddingConfig/LLMConfig - two places defining the
  same setting, with different names and different defaults. This
  composes those existing dataclasses instead, so there is one place
  that owns each setting's meaning and default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ragkit.config.chunker_config import ChunkerConfig
from ragkit.config.embedding_config import EmbeddingConfig
from ragkit.config.llm_config import LLMConfig


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """
    Configuration used by the RAGKit server.
    """

    vector_db_path: Path
    collection_name: str

    embedding: EmbeddingConfig
    llm: LLMConfig
    chunker: ChunkerConfig

    #
    # Final number of results returned after Reciprocal Rank Fusion
    # (see RAGService.top_k). Despite the shared name, this is NOT the
    # same setting as RetrievalConfig.top_k (SimilarityRetriever's own
    # fallback default) or RerankerConfig.top_k (post-rerank cutoff) -
    # the server doesn't use either of those, it calls RAGService with
    # this value explicitly. Kept as a plain field rather than forced
    # into RetrievalConfig to avoid implying a connection that doesn't
    # exist.
    #
    retrieval_top_k: int = 15


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
        embedding=EmbeddingConfig(
            model="nomic-embed-text",
        ),

        llm=LLMConfig(
            model="qwen3:8b",

            # Reasoning models (qwen3, deepseek-r1, ...) generate a
            # long "thinking" trace before the final answer, which is
            # slow on CPU-only machines but tends to produce more
            # thorough answers. Default to True for answer quality;
            # set RAGKIT_LLM_THINK=false to trade answer quality for
            # latency.
            think=_env_bool("RAGKIT_LLM_THINK", True),
        ),

        # Larger chunks give the LLM more context per retrieved source,
        # at the cost of retrieval precision. ChunkerConfig's own
        # library default (500/50) was too small for this app - it
        # produced terse, thin answers compared to bigger chunk sizes.
        # Only affects documents indexed AFTER this is applied; existing
        # documents must be re-uploaded to be re-chunked.
        chunker=ChunkerConfig(
            chunk_size=_env_int("RAGKIT_CHUNK_SIZE", 2500),
            overlap=_env_int("RAGKIT_CHUNK_OVERLAP", 300),
        ),

        retrieval_top_k=_env_int("RAGKIT_TOP_K", 15),
    )
