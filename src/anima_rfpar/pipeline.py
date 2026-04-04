from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import AttackConfig
from .types import AttackRunSummary


class RFPARPipeline:
    """Scaffold pipeline for RFPAR execution.

    This class intentionally provides a dry-run implementation first,
    so PRD-02/03 tasks can wire reference parity safely before real execution.
    """

    def __init__(self, config: AttackConfig):
        self.config = config

    def run(self) -> AttackRunSummary:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = AttackRunSummary(
            mode=self.config.mode,
            backend=self.config.backend,
            dry_run=self.config.dry_run,
            query_budget=self.config.query_budget,
            output_dir=output_dir,
        )

        metadata_path = output_dir / "run_metadata.json"
        metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "module": self.config.module_name,
            "summary": {
                "mode": summary.mode,
                "backend": summary.backend,
                "dry_run": summary.dry_run,
                "query_budget": summary.query_budget,
                "output_dir": str(summary.output_dir),
            },
            "config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
                "paths": {
                    "reference_repo": str(self.config.paths.reference_repo),
                    "imagenet_root": str(self.config.paths.imagenet_root),
                    "coco_root": str(self.config.paths.coco_root),
                    "argoverse_root": str(self.config.paths.argoverse_root),
                },
            },
            "note": "Scaffold dry-run only. Implement algorithm execution in PRD-02/03.",
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return summary
