from anima_rfpar.config import load_attack_config
from anima_rfpar.pipeline import RFPARPipeline


def test_pipeline_run_returns_summary() -> None:
    cfg = load_attack_config("configs/debug.toml")
    summary = RFPARPipeline(cfg).run()
    assert summary.mode in {"classification", "detection"}
    assert summary.backend in {"cpu", "cuda", "mlx"}
    assert summary.query_budget > 0
