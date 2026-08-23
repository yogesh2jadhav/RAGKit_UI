"""
Purpose
-------
Stores embeddings inside a local Chroma database.

Responsibilities
----------------
- Create/Open a Chroma collection.
- Persist embeddings.
- Persist complete Chunk metadata.

Does NOT
--------
- Generate embeddings.
- Perform similarity search.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from uuid import UUID

import chromadb

from ragkit.config.vector_store_config import VectorStoreConfig
from ragkit.exceptions.vector_store_error import VectorStoreError
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.search_result import SearchResult
from ragkit.vectorstores.vector_store import VectorStore

'''
=> ChromaVectorStore class implement VectorStore Interface and it's 3 methods.
'''
class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of VectorStore.
    """

    #
    # Metadata keys stored inside Chroma.
    #
    # Using constants prevents accidental typos and keeps
    # the storage and retrieval logic synchronized.
    #
    _MODEL = "_ragkit_model"
    _DOCUMENT_ID = "_ragkit_document_id"
    _CHUNK_INDEX = "_ragkit_chunk_index"
    _START_OFFSET = "_ragkit_start_offset"
    _END_OFFSET = "_ragkit_end_offset"
    '''
    => Following constructor will create connection to vector DB.
    '''

    def __init__(
        self,
        path: str = "./vector_db",
        collection_name: str = "ragkit",
        *,
        config: VectorStoreConfig | None = None,
    ) -> None:
        """
        Initialize the Chroma vector store.
        """

        #
        # Configuration overrides explicit parameters.
        #
        if config is not None:
            path = config.path
            collection_name = config.collection_name

        self._client = chromadb.PersistentClient(
            path=path,
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

    '''
    Following method add() will save embeddings in vector DB.
    And need two inputs Chunk and embedding.
    '''
    def add(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Embedding],
    ) -> None:
        """
        => embeddings: The source data. It is a collection or stream containing
        your Embedding objects.
        for embedding in embeddings: A standard loop that goes through each
        individual Embedding item one by one.
        embedding.chunk_id (The Key): This becomes the dictionary key. It is
        the identifier of the text chunk.
        embedding (The Value): This becomes the dictionary value. It is the
        full object itself (containing the vector array and the model metadata).

        embedding_by_chunk_id(chunk_id, embedding)
        """
        embedding_by_chunk_id = {
            #=> embedding_by_chunk_id(chunk_id, embedding)
            embedding.chunk_id: embedding for embedding in embeddings
        }

        for chunk in chunks:
            embedding = embedding_by_chunk_id.get(chunk.id)
            if embedding is None:
                raise VectorStoreError(f"No embedding found for chunk {chunk.id}.")

            metadata = {
                self._MODEL: embedding.model,
                self._DOCUMENT_ID: str(chunk.document_id),
                self._CHUNK_INDEX: chunk.index,
                self._START_OFFSET: chunk.start_offset,
                self._END_OFFSET: chunk.end_offset,
                **chunk.metadata,
            }

            self._collection.add( # => This method will push everything into vector db
                ids=[str(chunk.id)],
                embeddings=[embedding.vector],
                documents=[chunk.content],
                metadatas=[metadata],
            )


    def count(self) -> int:
        """
        Returns the number of stored vectors.
        """
        return self._collection.count()

    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> Iterable[SearchResult]:
        """
        Search for chunks similar to the supplied query.
        filters
            Optional metadata filters applied before
            similarity search.
        Responsibilities
        ----------------
        - Execute a similarity search in ChromaDB.
        - Reconstruct Chunk objects.
        - Yield SearchResult objects.

        Does NOT
        --------
        - Generate embeddings.
        - Rank or rerank results.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        '''
        => Following code will do query on vector DB and will get documents,
        metadata, and distances as retrun.
        '''
        response = self._collection.query(
            query_embeddings=[query_embedding.vector],
            n_results=top_k,
            where=self._build_where_clause(filters),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        '''
        => We get multiple respone from vector DB. but we will go with only index 0 for now.
        '''
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        #
        # All returned collections must have the same size.
        #
        assert len(ids) == len(documents) == len(metadatas) == len(distances)

        # => following for loop is just to create chunk object using output of
        # query and return as SearchResult object.
        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            chunk = self._build_chunk(
                chunk_id=chunk_id,
                document=document,
                metadata=metadata,
            )

            yield SearchResult(
                chunk=chunk,
                score=distance,
            )

    def _build_chunk(
        self,
        *,
        chunk_id: str,
        document: str,
        metadata: Mapping[str, Any],
    ) -> Chunk:
        """
        Reconstruct a Chunk from data stored in Chroma.
        Remove internal metadata fields.
        Everything remaining belongs to the original chunk.
        """
        custom_metadata = dict(metadata)
        custom_metadata.pop(self._MODEL, None)
        document_id = UUID(custom_metadata.pop(self._DOCUMENT_ID))
        index = custom_metadata.pop(self._CHUNK_INDEX)
        start_offset = custom_metadata.pop(self._START_OFFSET)
        end_offset = custom_metadata.pop(self._END_OFFSET)

        return Chunk(
            id=UUID(chunk_id),
            document_id=document_id,
            index=index,
            content=document,
            start_offset=start_offset,
            end_offset=end_offset,
            metadata=custom_metadata,
        )

    def iter_chunks(
        self,
    ) -> Iterable[Chunk]:
        """
        Iterate over every stored chunk.

        Responsibilities
        ----------------
        - Read all chunks from ChromaDB.
        - Reconstruct Chunk objects.
        - Yield chunks.

        Does NOT
        --------
        - Return embeddings.
        - Perform similarity search.
        """

        response = self._collection.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

        ids = response["ids"]
        documents = response["documents"]
        metadatas = response["metadatas"]

        assert len(ids) == len(documents) == len(metadatas)

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
            strict=True,
        ):
            yield self._build_chunk(
                chunk_id=chunk_id,
                document=document,
                metadata=metadata,
            )

    @staticmethod
    def _build_where_clause(
        filters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Convert RAGKit filters into a Chroma where clause.
        """

        if not filters:
            return None

        if len(filters) == 1:
            return dict(filters)

        return {"$and": [{key: value} for key, value in filters.items()]}

    def clear(self) -> None:
        """
        Remove all indexed chunks from the collection.
        """
        ids = self._collection.get()["ids"]

        if ids:
            self._collection.delete(
                ids=ids,
            )


    def delete_document(
        self,
        document_id: UUID,
    ) -> None:
        """
        Remove all chunks belonging to a document.

        The document ID is stored in Chroma metadata under
        the internal RAGKit document ID field.
        """

        response = self._collection.get(
            where={
                self._DOCUMENT_ID: str(document_id),
            },
        )

        ids = response["ids"]

        if ids:
            self._collection.delete(
                ids=ids,
            )