"""
Purpose
-------
Represents the original and normalized versions of a user query.

Responsibilities
----------------
- Preserve the user's original question.
- Store the query used for retrieval.

Does NOT
--------
- Perform query normalization.
- Perform retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedQuery:
    """
    Result of query normalization.
    """

    original: str
    normalized: str