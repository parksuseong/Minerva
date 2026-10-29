from fastapi.testclient import TestClient

from minerva.main import app

client = TestClient(app)


def test_index_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Minerva" in response.text


def test_index_has_theme_switch():
    page = client.get("/").text
    assert 'id="theme"' in page
    for mode in ("system", "light", "dark"):
        assert f'data-theme="{mode}"' in page
    assert ':root[data-theme="dark"]' in page


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "server_time" in body


def test_pages_revalidate_so_ui_changes_show():
    from fastapi.testclient import TestClient

    from minerva.main import app

    client = TestClient(app)
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/static/office.css").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/health").headers
