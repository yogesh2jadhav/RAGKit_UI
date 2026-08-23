from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from ragkit.api.app import create_app
from ragkit.models.document_info import DocumentInfo


class FakeDocumentService:
    """
    Fake DocumentService used by API tests.
    """

    def __init__(self) -> None:
        self.document_id = uuid4()
        self.deleted_document_id: UUID | None = None

    def list_documents(self) -> list[DocumentInfo]:
        """
        Return fake indexed documents.
        """

        return [
            DocumentInfo(
                id=self.document_id,
                filename="Yogesh Ashok 007.docx",
                chunk_count=10,
            ),
            DocumentInfo(
                id=uuid4(),
                filename="Resume.docx",
                chunk_count=5,
            ),
        ]

    def delete_document(
        self,
        *,
        document_id: UUID,
    ) -> None:
        """
        Record the document deletion request.
        """

        self.deleted_document_id = document_id


def create_test_app(
    monkeypatch,
    fake_service: FakeDocumentService,
):
    """
    Create an application using the fake document service.
    """

    from ragkit.api import app as app_module

    monkeypatch.setattr(
        app_module,
        "DocumentService",
        lambda **kwargs: fake_service,
    )

    return create_app()


def test_list_documents_endpoint(monkeypatch):
    """
    Verify GET /api/documents returns indexed documents.
    """

    fake_service = FakeDocumentService()

    app = create_test_app(
        monkeypatch,
        fake_service,
    )

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


def test_delete_document_endpoint(monkeypatch):
    """
    Verify DELETE /api/documents/{document_id}
    passes the document ID to DocumentService.
    """

    fake_service = FakeDocumentService()

    app = create_test_app(
        monkeypatch,
        fake_service,
    )

    client = TestClient(app)

    document_id = fake_service.document_id

    response = client.delete(
        f"/api/documents/{document_id}",
    )

    assert response.status_code == 204

    assert fake_service.deleted_document_id == document_id


def test_delete_document_rejects_invalid_uuid(monkeypatch):
    """
    Verify DELETE rejects an invalid document ID.
    """

    fake_service = FakeDocumentService()

    app = create_test_app(
        monkeypatch,
        fake_service,
    )

    client = TestClient(app)

    response = client.delete(
        "/api/documents/not-a-uuid",
    )

    assert response.status_code == 422

    assert fake_service.deleted_document_id is None