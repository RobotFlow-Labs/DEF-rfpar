from anima_rfpar.service.app import health, ready


def test_health_contract() -> None:
    resp = health()
    assert resp["status"] == "ok"


def test_ready_contract() -> None:
    resp = ready()
    assert resp["status"] == "ready"
