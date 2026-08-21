"""
Purpose
-------
Represents information about an indexed document.

Responsibilities
----------------
- Identify an indexed document.
- Expose the display filename.
- Report the number of indexed chunks.

Does NOT
--------
- Load documents.
- Generate embeddings.
- Store documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    """
    Information about an indexed document.
    """

    id: UUID
    filename: str
    chunk_count: int