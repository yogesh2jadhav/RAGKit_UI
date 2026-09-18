"""
Purpose
-------
Configuration for rerankers.

Responsibilities
----------------
- Configure reranking behaviour.
- Store reranker-specific options.

Does NOT
--------
- Perform reranking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """
    Configuration for rerankers.
    """

    #
    # Number of results to return
    # after reranking.
    #
    top_k: int = 15

    #
    # Temperature used by LLM based
    # rerankers.
    #
    temperature: float = 0.0

    #
    # Maximum tokens generated while
    # producing the ranking.
    #
    max_tokens: int = 64