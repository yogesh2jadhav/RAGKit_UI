from uuid import uuid4

from fastapi.testclient import TestClient

from ragkit.api.app import create_app
from ragkit.models.rag_response import RAGResponse, RAGSource


class FakeRAGService:
    """
    Fake RAGService used by API tests.
    """

    def __init__(self) -> None:
        self.last_query = None
        self.last_document_ids = None

    def ask(
        self,
        query: str,
        *,
        document_ids=None,
        filters=None,
    ) -> RAGResponse:
        """
        Return a predictable RAG response.
        """

        self.last_query = query
        self.last_document_ids = document_ids

        document_id = uuid4()
        chunk_id = uuid4()

        return RAGResponse(
            answer="TEST ANSWER",
            sources=[
                RAGSource(
                    document_id=document_id,
                    filename="Yogesh Ashok 007.docx",
                    chunk_id=chunk_id,
                    score=0.95,
                )
            ],
        )


def test_chat_endpoint_returns_answer_and_sources():
    """
    Verify POST /api/chat returns the RAG answer and sources.
    """

    rag_service = FakeRAGService()

    app = create_app(
        rag_service=rag_service,
    )

    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "question": "How many years of experience?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "TEST ANSWER"

    assert len(data["sources"]) == 1

    assert (
        data["sources"][0]["filename"]
        == "Yogesh Ashok 007.docx"
    )


def test_chat_endpoint_passes_document_ids():
    """
    Verify selected document IDs are passed to RAGService.
    """

    rag_service = FakeRAGService()

    app = create_app(
        rag_service=rag_service,
    )

    client = TestClient(app)

    document_id = uuid4()

    response = client.post(
        "/api/chat",
        json={
            "question": "How many years of experience?",
            "document_ids": [
                str(document_id),
            ],
        },
    )

    assert response.status_code == 200

    assert rag_service.last_query == (
        "How many years of experience?"
    )

    assert rag_service.last_document_ids == [
        document_id,
    ]


def test_chat_endpoint_accepts_multiple_document_ids():
    """
    Verify multiple selected documents are accepted.
    """

    rag_service = FakeRAGService()

    app = create_app(
        rag_service=rag_service,
    )

    client = TestClient(app)

    document_id_1 = uuid4()
    document_id_2 = uuid4()

    response = client.post(
        "/api/chat",
        json={
            "question": "Who has Spark experience?",
            "document_ids": [
                str(document_id_1),
                str(document_id_2),
            ],
        },
    )

    assert response.status_code == 200

    assert rag_service.last_document_ids == [
        document_id_1,
        document_id_2,
    ]


def test_chat_endpoint_rejects_empty_question():
    """
    Verify an empty question is rejected by the API.
    """

    rag_service = FakeRAGService()

    app = create_app(
        rag_service=rag_service,
    )

    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422