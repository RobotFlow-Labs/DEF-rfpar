from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import BenchmarkAggregate
from .config import load_attack_config
from .pipeline import RFPARPipeline


def _cmd_check_assets(args: argparse.Namespace) -> int:
    assets = Path(args.assets)
    required = [
        Path("papers/2502.07821.pdf"),
        Path("repositories/RFPAR/main_cls.py"),
        Path("repositories/RFPAR/main_od.py"),
    ]

    report = {
        "assets_manifest_exists": assets.exists(),
        "required_files": {str(p): p.exists() for p in required},
    }
    print(json.dumps(report, indent=2))
    return 0


def _cmd_plan_benchmark(_: argparse.Namespace) -> int:
    template = BenchmarkAggregate(
        n=0,
        success_rate=0.0,
        l0_mean=0.0,
        l2_mean=0.0,
        query_mean=0.0,
        rd_mean=0.0,
        rm_mean=0.0,
        ata_mean=0.0,
        latency_ms_mean=0.0,
        memory_mb_mean=0.0,
    )
    print(template)
    return 0


def _cmd_attack(args: argparse.Namespace) -> int:
    cfg = load_attack_config(args.config)
    pipeline = RFPARPipeline(cfg)
    summary = pipeline.run()
    print(
        json.dumps(
            {
                "mode": summary.mode,
                "backend": summary.backend,
                "dry_run": summary.dry_run,
                "query_budget": summary.query_budget,
                "output_dir": str(summary.output_dir),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anima-rfpar", description="DEF-rfpar scaffold CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_assets = sub.add_parser("check-assets", help="Validate local paper/reference assets")
    p_assets.add_argument("--assets", default="ASSETS.md")
    p_assets.set_defaults(func=_cmd_check_assets)

    p_plan = sub.add_parser("plan-benchmark", help="Print benchmark schema template")
    p_plan.set_defaults(func=_cmd_plan_benchmark)

    p_attack = sub.add_parser("attack", help="Run pipeline in scaffold mode")
    p_attack.add_argument("--config", default="configs/default.toml")
    p_attack.set_defaults(func=_cmd_attack)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
