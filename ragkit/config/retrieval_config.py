"""
Purpose
-------
Configuration for document retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

# => This is just like DTO in java

@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """
    Retrieval configuration.
    """

    top_k: int = 15