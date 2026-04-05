from anima_rfpar.service.app import app, health, node, ready


def test_health_contract() -> None:
    resp = health()
    assert resp["status"] == "ok"
    assert resp["module"] == "DEF-rfpar"


def test_ready_contract() -> None:
    resp = ready()
    if hasattr(resp, "body"):
        import json

        data = json.loads(resp.body)
        assert data["ready"] is False
    else:
        assert "ready" in resp
