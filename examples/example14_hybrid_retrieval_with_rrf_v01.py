"""
Example 14 - Hybrid Retrieval with Reciprocal Rank Fusion
Version : v01

This example demonstrates how semantic search and BM25 keyword
search can be combined using Reciprocal Rank Fusion (RRF).

Learning Objectives
-------------------
- Compare semantic retrieval with BM25 retrieval.
- Understand why the two retrieval strategies can produce
  different rankings.
- Understand how RRF combines ranked result lists.
- See the final ranking produced by HybridRetriever.

RRF Formula
-----------
    RRF Score = 1 / (k + rank)

where:

    k    = RRF constant, normally 60
    rank = position of the result in a ranked list

Important
---------
RRF does NOT compare semantic scores with BM25 scores.

It uses the rank position from each retriever.

Example:

Semantic Search
    1. A
    2. B
    3. C

BM25 Search
    1. C
    2. D
    3. A

RRF rewards results that appear near the top of multiple
retrieval lists.
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.retrievers.hybrid_retriever import HybridRetriever
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.sources.local_source import LocalSource
from ragkit.transformers.markdown_transformer import MarkdownTransformer
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


EXAMPLES_DIR = Path(__file__).parent

DOCS_DIR = EXAMPLES_DIR / "docs"

DATA_DIR = EXAMPLES_DIR / "data"

VECTOR_DB_DIR = DATA_DIR / "vector_db_rrf"

COLLECTION_NAME = "ragkit_rrf"


def build_index(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Build the vector index from the example documents.

    The same indexed chunks are later used by both:

    - SimilarityRetriever
    - BM25Searcher
    """

    processor = DocumentProcessor(
        transformer=MarkdownTransformer(),
        chunker=CharacterChunker(
            chunk_size=300,
            chunk_overlap=50,
        ),
        embedder=OllamaEmbedder(
            model="nomic-embed-text",
        ),
    )

    indexer = DocumentIndexer(
        processor=processor,
        vector_store=vector_store,
    )

    source = LocalSource(
        directory=DOCS_DIR,
    )

    result = indexer.index(
        source,
    )

    print()
    print("=" * 70)
    print("Index Build Completed")
    print("=" * 70)
    print(result)
    print()


def print_results(
    title: str,
    results,
) -> None:
    """
    Print retrieved results.

    Note
    ----
    The score displayed here is the score returned by the
    underlying retriever.

    For semantic search this is the vector-store score.

    For BM25 this is the BM25 score.

    After RRF, the important thing is the final ORDER.
    We intentionally do not label the underlying score
    as an RRF score.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    results = list(results)

    if not results:
        print("No results found.")
        print()
        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        content = result.chunk.content.replace(
            "\n",
            " ",
        )

        metadata = result.chunk.metadata

        print(
            f"{index}. Score : {result.score:.4f}",
        )

        print(
            "   File    :",
            metadata.get(
                "filename",
                "-",
            ),
        )

        print(
            "   Content :",
            content[:120],
        )

        print()


def explain_rrf() -> None:
    """
    Explain the RRF calculation used by the hybrid retriever.
    """

    print()
    print("=" * 70)
    print("How Reciprocal Rank Fusion Works")
    print("=" * 70)

    print(
        """
RRF does not compare the scores produced by semantic search
and BM25.

Instead, it uses the RANK of each result.

Formula:

    RRF Score = 1 / (k + rank)

Default:

    k = 60

Example:

Semantic Search
    Rank 1 -> Chunk A
    Rank 2 -> Chunk B
    Rank 3 -> Chunk C

BM25 Search
    Rank 1 -> Chunk C
    Rank 2 -> Chunk D
    Rank 3 -> Chunk A

Chunk A receives:

    1 / (60 + 1)
  + 1 / (60 + 3)

Chunk C receives:

    1 / (60 + 3)
  + 1 / (60 + 1)

Therefore, a chunk that appears near the top of BOTH
retrieval lists receives contributions from both lists.

This is the key idea behind RRF.
""",
    )


def compare(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Compare semantic, BM25 and RRF hybrid retrieval.
    """

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
    )

    #
    # Semantic retrieval.
    #
    semantic_retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    #
    # Keyword retrieval using BM25.
    #
    keyword_searcher = BM25Searcher(
        vector_store=vector_store,
    )

    #
    # HybridRetriever now uses ReciprocalRankFusion
    # internally by default.
    #
    hybrid_retriever = HybridRetriever(
        retriever=semantic_retriever,
        keyword_searcher=keyword_searcher,
    )

    while True:

        print()

        query = input(
            "Query (back/exit/quit): ",
        ).strip()

        if query.lower() in {
            "back",
            "exit",
            "quit",
        }:
            break

        if not query:
            print("Please enter a query.")
            continue

        #
        # Retrieve independently from semantic search.
        #
        semantic_results = semantic_retriever.retrieve(
            query=query,
            top_k=5,
        )

        #
        # Retrieve independently from BM25.
        #
        keyword_results = keyword_searcher.search(
            query=query,
            top_k=5,
        )

        #
        # Retrieve using HybridRetriever.
        #
        # HybridRetriever sends both ranked lists to
        # ReciprocalRankFusion and returns the fused ranking.
        #
        hybrid_results = hybrid_retriever.retrieve(
            query=query,
            top_k=5,
        )

        print_results(
            "1. Semantic Search",
            semantic_results,
        )

        print_results(
            "2. BM25 Keyword Search",
            keyword_results,
        )

        print_results(
            "3. Hybrid Search with RRF",
            hybrid_results,
        )

        explain_rrf()


def statistics(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Display vector-store statistics.
    """

    print()
    print("=" * 70)
    print("Statistics")
    print("=" * 70)

    print(
        "Vectors :",
        vector_store.count(),
    )

    print()


def main() -> None:
    """
    Run Example 14.
    """

    print("=" * 70)
    print("Example 14 - Hybrid Retrieval with RRF")
    print("Version v01")
    print("=" * 70)

    vector_store = ChromaVectorStore(
        path=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    while True:

        print()
        print("1. Build Index")
        print("2. Compare Retrieval")
        print("3. Statistics")
        print("4. Exit")

        choice = input(
            "Select: ",
        ).strip()

        if choice == "1":

            build_index(
                vector_store,
            )

        elif choice == "2":

            compare(
                vector_store,
            )

        elif choice == "3":

            statistics(
                vector_store,
            )

        elif choice == "4":

            break

        else:

            print("Invalid selection.")


if __name__ == "__main__":
    main()