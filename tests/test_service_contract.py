from anima_rfpar.service.app import health, ready


def test_health_contract() -> None:
    resp = health()
    assert resp["status"] == "ok"
    assert resp["module"] == "DEF-rfpar"


def test_ready_contract() -> None:
    resp = ready()
    # Without weights loaded, returns JSONResponse (503) or dict
    if hasattr(resp, "body"):
        # JSONResponse case — weights not loaded
        import json

        data = json.loads(resp.body)
        assert data["ready"] is False
    else:
        assert "ready" in resp
