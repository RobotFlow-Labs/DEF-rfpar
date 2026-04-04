import json

from anima_rfpar.config import load_attack_config
from anima_rfpar.pipeline import RFPARPipeline


def test_attack_smoke_writes_metadata() -> None:
    cfg = load_attack_config("configs/debug.toml")
    summary = RFPARPipeline(cfg).run()
    metadata_path = summary.output_dir / "run_metadata.json"
    assert metadata_path.exists()
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["summary"]["mode"] in {"classification", "detection"}
