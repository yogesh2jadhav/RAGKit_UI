from uuid import uuid4

from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def create_chunk(
    content: str,
    index: int = 0,
    filename: str = "test.docx",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=index,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={
            "filename": filename,
        },
    )


def create_embedding(
    chunk: Chunk,
) -> Embedding:

    #
    # BM25 never uses embeddings,
    # but Chroma requires one.
    #
    return Embedding(
        chunk_id=chunk.id,
        model="unit-test",
        vector=[0.1, 0.2, 0.3],
    )


def test_bm25_returns_best_match(
    tmp_path,
):
    """
    Verify BM25 returns the best keyword match.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunks = [
        create_chunk("Apache Spark is fast"),
        create_chunk("Python is easy"),
        create_chunk("Delta Lake supports ACID"),
    ]

    store.add(
        chunks=chunks,
        embeddings=[
            create_embedding(chunk)
            for chunk in chunks
        ],
    )

    searcher = BM25Searcher(
        vector_store=store,
    )

    results = list(
        searcher.search(
            "spark",
        )
    )

    assert len(results) > 0

    assert (
        results[0]
        .chunk
        .content
        == "Apache Spark is fast"
    )


def test_bm25_respects_top_k(
    tmp_path,
):
    """
    Verify top_k limits results.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunks = [
        create_chunk(f"Spark document {i}")
        for i in range(5)
    ]

    store.add(
        chunks=chunks,
        embeddings=[
            create_embedding(chunk)
            for chunk in chunks
        ],
    )

    searcher = BM25Searcher(
        vector_store=store,
    )

    results = list(
        searcher.search(
            "spark",
            top_k=2,
        )
    )

    assert len(results) == 2


def test_bm25_unknown_query(
    tmp_path,
):
    """
    Verify unknown queries do not fail.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = create_chunk(
        "Apache Spark",
    )

    store.add(
        chunks=[chunk],
        embeddings=[
            create_embedding(chunk)
        ],
    )

    searcher = BM25Searcher(
        vector_store=store,
    )

    results = list(
        searcher.search(
            "abcdefghijk",
        )
    )

    #
    # BM25 returns a score even if it is zero.
    #
    assert len(results) == 1

    def create_chunk_for_document(
        content: str,
        document_id,
        index: int = 0,
    ) -> Chunk:
        return Chunk(
            id=uuid4(),
            document_id=document_id,
            index=index,
            content=content,
            start_offset=0,
            end_offset=len(content),
            metadata={},
        )

    def test_bm25_filters_by_document_id(
            tmp_path,
    ):
        """
        Verify BM25 only returns chunks from
        the selected document.
        """

        from uuid import uuid4

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        document_a = uuid4()
        document_b = uuid4()

        chunk_a = create_chunk_for_document(
            "Apache Spark is fast",
            document_a,
        )

        chunk_b = create_chunk_for_document(
            "Apache Spark runs on Kubernetes",
            document_b,
        )

        store.add(
            chunks=[
                chunk_a,
                chunk_b,
            ],
            embeddings=[
                create_embedding(chunk_a),
                create_embedding(chunk_b),
            ],
        )

        searcher = BM25Searcher(
            vector_store=store,
        )

        results = list(
            searcher.search(
                "spark",
                document_ids=[document_a],
            )
        )

        assert len(results) == 1

        assert results[0].chunk.document_id == document_a
        assert results[0].chunk.content == "Apache Spark is fast"

    def test_bm25_filters_by_document_id(
            tmp_path,
    ):
        """
        Verify BM25 only returns chunks from
        the selected document.
        """

        from uuid import uuid4

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        document_a = uuid4()
        document_b = uuid4()

        chunk_a = create_chunk_for_document(
            "Apache Spark is fast",
            document_a,
        )

        chunk_b = create_chunk_for_document(
            "Apache Spark runs on Kubernetes",
            document_b,
        )

        store.add(
            chunks=[
                chunk_a,
                chunk_b,
            ],
            embeddings=[
                create_embedding(chunk_a),
                create_embedding(chunk_b),
            ],
        )

        searcher = BM25Searcher(
            vector_store=store,
        )

        results = list(
            searcher.search(
                "spark",
                document_ids=[document_a],
            )
        )

        assert len(results) == 1

        assert results[0].chunk.document_id == document_a
        assert results[0].chunk.content == "Apache Spark is fast"

    def test_bm25_filters_multiple_documents(
            tmp_path,
    ):
        """
        Verify BM25 can search multiple selected documents.
        """

        from uuid import uuid4

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        document_a = uuid4()
        document_b = uuid4()
        document_c = uuid4()

        chunks = [
            create_chunk_for_document(
                "Apache Spark A",
                document_a,
            ),
            create_chunk_for_document(
                "Apache Spark B",
                document_b,
            ),
            create_chunk_for_document(
                "Apache Spark C",
                document_c,
            ),
        ]

        store.add(
            chunks=chunks,
            embeddings=[
                create_embedding(chunk)
                for chunk in chunks
            ],
        )

        searcher = BM25Searcher(
            vector_store=store,
        )

        results = list(
            searcher.search(
                "spark",
                document_ids=[
                    document_a,
                    document_b,
                ],
            )
        )

        assert len(results) == 2

        result_document_ids = {
            result.chunk.document_id
            for result in results
        }

        assert result_document_ids == {
            document_a,
            document_b,
        }

    def test_bm25_searches_filename_metadata(
            tmp_path,
    ):
        """
        Verify BM25 searches document filename metadata
        in addition to chunk content.
        """

        store = ChromaVectorStore(
            path=str(tmp_path),
            collection_name="unit_test",
        )

        yogesh_chunk = create_chunk(
            "The profile describes nearly 20 years of rich expertise.",
            filename="Yogesh Ashok 007.docx",
        )

        ashish_chunk = create_chunk(
            "The profile describes professional experience.",
            filename="Ashish Pawar (1).docx",
        )

        store.add(
            chunks=[
                yogesh_chunk,
                ashish_chunk,
            ],
            embeddings=[
                create_embedding(yogesh_chunk),
                create_embedding(ashish_chunk),
            ],
        )

        searcher = BM25Searcher(
            vector_store=store,
        )

        results = list(
            searcher.search(
                "How many years of experience does Yogesh have?",
                top_k=2,
            )
        )

        assert len(results) == 2

        assert results[0].chunk.document_id == (
            yogesh_chunk.document_id
        )

        assert results[0].chunk.metadata["filename"] == (
            "Yogesh Ashok 007.docx"
        )