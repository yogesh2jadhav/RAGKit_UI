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
from time import sleep

from ragkit.chunkers.structured_text_chunker import StructuredTextChunker
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
from ragkit.services.rag_service import RAGService

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
        chunker=StructuredTextChunker(
            chunk_size=1000,
            chunk_overlap=100,
        ),
        embedder=OllamaEmbedder(
            model=EMBEDDING_MODEL,
        ),
    )

    indexer = DocumentIndexer(
        processor=processor,
        vector_store=vector_store,
    )

    print("Clearing existing index...")

    vector_store.clear()

    print("Building fresh index...")

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

    RAG orchestration is handled by RAGService.

    The CLI is responsible only for:
    - Reading user input.
    - Displaying the answer.
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

    #
    # Application-level RAG service.
    #
    rag_service = RAGService(
        retriever=vector_retriever,
        keyword_searcher=keyword_searcher,
        rrf=rrf,
        prompt_builder=prompt_builder,
        llm=llm,
        top_k=RETRIEVAL_TOP_K,
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
        # RAGService now handles:
        #
        # Vector Search
        # BM25
        # RRF
        # Prompt Builder
        # LLM
        #
        response = rag_service.ask(query)

        print()
        print("=" * 70)
        print("Assistant")
        print("=" * 70)
        print()
        print(response.answer)

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