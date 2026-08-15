"""
Example 15
----------

RRF-based Retrieval-Augmented Generation chat.

Concepts
--------
- Document indexing
- Vector similarity search
- BM25 keyword search
- Reciprocal Rank Fusion (RRF)
- Prompt construction
- Local LLM generation
- Interactive chat

Architecture
------------

                    User Question
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Vector Search               BM25 Search
              |                       |
              +-----------+-----------+
                          |
                          v
                 Reciprocal Rank
                    Fusion
                          |
                          v
                  Retrieved Context
                          |
                          v
                   Prompt Builder
                          |
                          v
                     Ollama LLM
                          |
                          v
                     Answer

Run
---
python examples/15_rrf_chat.py
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.llms.ollama_llm import OllamaLLM
from ragkit.models.search_result import SearchResult
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.prompts.default_prompt_builder import DefaultPromptBuilder
from ragkit.ranking.reciprocal_rank_fusion import ReciprocalRankFusion
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.sources.local_source import LocalSource
from ragkit.transformers.markdown_transformer import MarkdownTransformer
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


EXAMPLES_DIR = Path(__file__).parent
DOCS_DIR = EXAMPLES_DIR / "docs"

DATA_DIR = EXAMPLES_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db_rrf"

COLLECTION_NAME = "ragkit_rrf"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"

RETRIEVAL_TOP_K = 2555


def build_index(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Build the vector index.

    The same indexed chunks will later be used by:

    - Vector Search
    - BM25 Search
    - RRF
    """

    print()
    print("=" * 70)
    print("Building RRF Chat Index")
    print("=" * 70)
    print()

    processor = DocumentProcessor(
        transformer=MarkdownTransformer(),
        chunker=CharacterChunker(
            chunk_size=300,
            chunk_overlap=50,
        ),
        embedder=OllamaEmbedder(
            model=EMBEDDING_MODEL,
        ),
    )

    indexer = DocumentIndexer(
        processor=processor,
        vector_store=vector_store,
    )

    result = indexer.index(
        LocalSource(DOCS_DIR),
    )

    print()
    print("Index completed.")
    print()
    print(f"Documents : {result.documents}")
    print(f"Chunks    : {result.chunks}")
    print(f"Vectors   : {result.embeddings}")


def print_results(
    title: str,
    results: list[SearchResult],
) -> None:
    """
    Print retrieval results for learning/debugging.

    Showing the intermediate results is intentional.

    It allows us to see whether a problem comes from:

    Vector Search
    BM25
    RRF
    Prompt
    LLM
    """

    #print()
    #print("-" * 70)
    #print(title)
    #print("-" * 70)

    if not results:
        print("No results found.")
        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        filename = result.chunk.metadata.get(
            "filename",
            "-",
        )

        content = result.chunk.content.replace(
            "\n",
            " ",
        )

       # print()
       # print(f"{index}. Score : {result.score:.6f}")
       # print(f"   File  : {filename}")
       # print(f"   Text  : {content[:180]}")


def chat(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Start interactive RRF-based RAG chat.

    Retrieval flow:

        Vector Search
              +
            BM25
              |
              v
             RRF
              |
              v
        Prompt Builder
              |
              v
            Ollama
    """

    #
    # Vector retriever.
    #
    vector_retriever = SimilarityRetriever(
        embedder=OllamaEmbedder(
            model=EMBEDDING_MODEL,
        ),
        vector_store=vector_store,
    )

    #
    # BM25 builds its lexical index from the
    # chunks already stored in the vector store.
    #
    keyword_searcher = BM25Searcher(
        vector_store=vector_store,
    )

    #
    # RRF combines the rankings from the
    # semantic and keyword searches.
    #
    rrf = ReciprocalRankFusion()

    #
    # Prompt builder converts the final RRF
    # results into the LLM prompt.
    #
    prompt_builder = DefaultPromptBuilder()

    #
    # Local LLM.
    #
    llm = OllamaLLM(
        model=LLM_MODEL,
    )

    print()
    print("=" * 70)
    print("RRF Chat")
    print("=" * 70)
    print()
    print("Enter 'back' to return to the main menu.")

    while True:

        print()

        query = input("You : ").strip()

        if query.lower() in {
            "back",
            "exit",
            "quit",
        }:
            break

        if not query:
            continue

        #
        # --------------------------------------------------------
        # 1. VECTOR SEARCH
        # --------------------------------------------------------
        #

        vector_results = list(
            vector_retriever.retrieve(
                query=query,
                top_k=RETRIEVAL_TOP_K,
            )
        )

        #
        # --------------------------------------------------------
        # 2. BM25 SEARCH
        # --------------------------------------------------------
        #

        bm25_results = list(
            keyword_searcher.search(
                query=query,
                top_k=RETRIEVAL_TOP_K,
            )
        )

        #
        # --------------------------------------------------------
        # 3. RRF FUSION
        # --------------------------------------------------------
        #

        rrf_results = rrf.fuse(
            [
                vector_results,
                bm25_results,
            ],
            top_k=RETRIEVAL_TOP_K,
        )

        #
        # Show the intermediate retrieval results.
        #
        print_results(
            "Vector Search Results",
            vector_results,
        )

        print_results(
            "BM25 Results",
            bm25_results,
        )

        print_results(
            "RRF Results",
            rrf_results,
        )

        #
        # --------------------------------------------------------
        # 4. BUILD LLM PROMPT
        # --------------------------------------------------------
        #

        prompt = prompt_builder.build(
            query=query,
            search_results=rrf_results,
        )

        #
        # --------------------------------------------------------
        # 5. CALL LOCAL LLM
        # --------------------------------------------------------
        #

        response = llm.generate(
            prompt=prompt,
        )

        print()
        print("=" * 70)
        print("Assistant")
        print("=" * 70)
        print()
        print(response.content)


def statistics(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Display vector-store statistics.
    """

    print()
    print("=" * 70)
    print("RRF Chat Statistics")
    print("=" * 70)
    print()

    print(
        "Stored Vectors :",
        vector_store.count(),
    )

    print(
        "Collection     :",
        COLLECTION_NAME,
    )

    print(
        "Database Path  :",
        VECTOR_DB_DIR,
    )


def main() -> None:
    """
    Main menu.
    """

    vector_store = ChromaVectorStore(
        path=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    while True:

        print()
        print("=" * 70)
        print("RAGKit - RRF Chat Example")
        print("=" * 70)

        print("1. Build Index")
        print("2. Chat")
        print("3. Statistics")
        print("4. Exit")

        print()

        choice = input(
            "Select option: ",
        ).strip()

        if choice == "1":

            build_index(
                vector_store,
            )

        elif choice == "2":

            chat(
                vector_store,
            )

        elif choice == "3":

            statistics(
                vector_store,
            )

        elif choice == "4":

            print()
            print("Goodbye.")

            break

        else:

            print()
            print("Invalid option.")


if __name__ == "__main__":
    main()