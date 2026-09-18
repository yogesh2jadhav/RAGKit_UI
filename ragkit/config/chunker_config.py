"""
Purpose
-------
Configuration for Chunker implementations.

Responsibilities
----------------
- Store chunking configuration.
- Provide sensible defaults.

Does NOT
--------
- Perform chunking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """
    Configuration for text chunking.
    """

    chunk_size: int = 2500

    overlap: int = 300

    def __post_init__(self) -> None:
        """
        Validate configuration.
        """

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if self.overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if self.overlap >= self.chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )