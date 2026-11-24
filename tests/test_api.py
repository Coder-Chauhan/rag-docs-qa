"""API tests that run fully offline using fake embeddings and a fake chat model."""

import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.main import app
from app.rag import RAGService

FAKE_ANSWER = "Refunds are available within 30 days of purchase [1]."


@pytest.fixture
def client():
    app.state.rag = RAGService(
        embeddings=DeterministicFakeEmbedding(size=64),
        llm=FakeListChatModel(responses=[FAKE_ANSWER]),
        persist_dir=None,  # in-memory store
        collection_name=f"test-{uuid.uuid4().hex}",
        chunk_size=200,
        chunk_overlap=20,
    )
    with TestClient(app) as c:
        yield c
    app.state.rag = None


def upload(client, name: str, content: bytes):
    return client.post("/documents", files={"file": (name, content, "text/plain")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_query_without_documents(client):
    res = client.post("/query", json={"question": "What is the refund policy?"})
    assert res.status_code == 200
    assert res.json()["sources"] == []
    assert "No documents" in res.json()["answer"]


def test_upload_and_query(client):
    text = b"Our refund policy: customers can request a refund within 30 days of purchase. " * 10
    res = upload(client, "policy.txt", text)
    assert res.status_code == 201
    body = res.json()
    assert body["filename"] == "policy.txt"
    assert body["chunks_added"] > 1

    res = client.post("/query", json={"question": "What is the refund policy?", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == FAKE_ANSWER
    assert len(data["sources"]) == 2
    assert data["sources"][0]["source"] == "policy.txt"


def test_list_and_reset(client):
    upload(client, "a.md", b"# Notes\nSome markdown content about onboarding.")
    upload(client, "b.txt", b"Plain text content about security training.")

    stats = client.get("/documents").json()
    assert stats["documents"] == ["a.md", "b.txt"]
    assert stats["total_chunks"] >= 2

    assert client.delete("/documents").status_code == 204
    assert client.get("/documents").json()["total_chunks"] == 0


def test_rejects_unsupported_file(client):
    res = upload(client, "image.png", b"\x89PNG")
    assert res.status_code == 415


def test_rejects_empty_file(client):
    res = upload(client, "empty.txt", b"   ")
    assert res.status_code == 422


def test_validates_question(client):
    res = client.post("/query", json={"question": "hi"})
    assert res.status_code == 422
