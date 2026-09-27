from fastapi.testclient import TestClient

from app.api.main import app


def test_healthz_never_500s():
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ok", "degraded"}


def test_query_rejects_too_short_question():
    with TestClient(app) as client:
        assert client.post("/query", json={"question": "a"}).status_code == 422


def test_figure_endpoint_blocks_path_traversal():
    with TestClient(app) as client:
        resp = client.get("/figures/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in {400, 404}
