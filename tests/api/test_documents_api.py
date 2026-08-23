from fastapi.testclient import TestClient

from ragkit.api.app import create_app
from ragkit.models.document_info import DocumentInfo
from ragkit.services.rag_service_factory import create_rag_service

class FakeDocumentService:
    """
    Fake DocumentService used by API tests.
    """

    def list_documents(self) -> list[DocumentInfo]:
        """
        Return fake indexed documents.
        """

        from uuid import uuid4

        return [
            DocumentInfo(
                id=uuid4(),
                filename="Yogesh Ashok 007.docx",
                chunk_count=10,
            ),
            DocumentInfo(
                id=uuid4(),
                filename="Resume.docx",
                chunk_count=5,
            ),
        ]


def test_list_documents_endpoint(monkeypatch):
    """
    Verify GET /api/documents returns indexed documents.
    """

    fake_service = FakeDocumentService()

    from ragkit.api import app as app_module

    monkeypatch.setattr(
        app_module,
        "DocumentService",
        lambda **kwargs: fake_service,
    )

    app = create_app()

    client = TestClient(app)

    response = client.get(
        "/api/documents",
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    assert data[0]["filename"] == "Yogesh Ashok 007.docx"
    assert data[0]["chunk_count"] == 10

    assert data[1]["filename"] == "Resume.docx"
    assert data[1]["chunk_count"] == 5