"""
Purpose
-------
Split structured text into logical, size-bounded chunks.

Responsibilities
----------------
- Preserve paragraph and blank-line boundaries.
- Keep related logical blocks together.
- Avoid splitting normal paragraphs unnecessarily.
- Fall back to character splitting for oversized blocks.
- Preserve document metadata and chunk offsets.

Does NOT
--------
- Parse DOCX files.
- Generate embeddings.
- Store chunks.
- Retrieve chunks.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from ragkit.chunkers.chunker import Chunker
from ragkit.models.chunk import Chunk
from ragkit.models.document import Document


class StructuredTextChunker(Chunker):
    """
    Chunker that prefers logical text boundaries over
    arbitrary character boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(
        self,
        document: Document,
    ) -> Iterable[Chunk]:
        """
        Split a document into logical, size-bounded chunks.
        """

        content = document.content

        if not content:
            return

        blocks = self._split_into_blocks(content)

        current_blocks: list[tuple[int, str]] = []
        current_length = 0

        index = 0

        for start, block in blocks:

            block_length = len(block)

            #
            # Oversized logical block.
            #
            # Flush the current logical chunk first,
            # then split the oversized block.
            #
            if block_length > self._chunk_size:

                if current_blocks:
                    yield self._create_chunk(
                        document=document,
                        blocks=current_blocks,
                        index=index,
                    )

                    index += 1

                    current_blocks = []
                    current_length = 0

                for chunk in self._split_large_block(
                    document=document,
                    start=start,
                    text=block,
                    index_start=index,
                ):
                    yield chunk
                    index += 1

                continue

            #
            # Check whether adding this block would
            # exceed the configured chunk size.
            #
            separator_length = (
                2 if current_blocks else 0
            )

            if (
                current_blocks
                and current_length
                + separator_length
                + block_length
                > self._chunk_size
            ):

                yield self._create_chunk(
                    document=document,
                    blocks=current_blocks,
                    index=index,
                )

                index += 1

                #
                # Start the next chunk.
                #
                current_blocks = []
                current_length = 0

            current_blocks.append(
                (start, block),
            )

            if current_length:
                current_length += 2

            current_length += block_length

        #
        # Flush final chunk.
        #
        if current_blocks:
            yield self._create_chunk(
                document=document,
                blocks=current_blocks,
                index=index,
            )

    @staticmethod
    def _split_into_blocks(
        content: str,
    ) -> list[tuple[int, str]]:
        """
        Split content using blank lines as logical boundaries.

        Returns
        -------
        list[tuple[int, str]]
            Each item contains:

            - start offset
            - cleaned block text
        """

        blocks: list[tuple[int, str]] = []

        position = 0

        for raw_block in content.split("\n\n"):

            block = raw_block.strip()

            if not block:
                position += len(raw_block) + 2
                continue

            raw_start = content.find(
                raw_block,
                position,
            )

            start = raw_start if raw_start >= 0 else position

            blocks.append(
                (
                    start,
                    block,
                )
            )

            position = start + len(raw_block)

        return blocks

    def _create_chunk(
        self,
        document: Document,
        blocks: list[tuple[int, str]],
        index: int,
    ) -> Chunk:
        """
        Create a Chunk from logical blocks.
        """

        content = "\n\n".join(
            block
            for _, block in blocks
        )

        start_offset = blocks[0][0]

        last_start, last_block = blocks[-1]

        end_offset = last_start + len(last_block)

        return Chunk(
            id=uuid4(),
            document_id=document.id,
            index=index,
            content=content,
            start_offset=start_offset,
            end_offset=end_offset,
            metadata=dict(document.metadata),
        )

    def _split_large_block(
        self,
        document: Document,
        start: int,
        text: str,
        index_start: int,
    ) -> Iterable[Chunk]:
        """
        Split an oversized logical block using character
        windows as a fallback.
        """

        position = 0
        index = index_start

        step = (
            self._chunk_size
            - self._chunk_overlap
        )

        while position < len(text):

            end = min(
                position + self._chunk_size,
                len(text),
            )

            chunk_content = text[position:end]

            yield Chunk(
                id=uuid4(),
                document_id=document.id,
                index=index,
                start_offset=start + position,
                end_offset=start + end,
                content=chunk_content,
                metadata=dict(document.metadata),
            )

            if end >= len(text):
                break

            position += step
            index += 1