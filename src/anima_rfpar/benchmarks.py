from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class BenchmarkSample:
    success: bool
    l0: float
    l2: float
    query_count: int
    rd: float = 0.0
    rm: float = 0.0
    ata: float = 0.0
    latency_ms: float = 0.0
    memory_mb: float = 0.0


@dataclass(frozen=True)
class BenchmarkAggregate:
    n: int
    success_rate: float
    l0_mean: float
    l2_mean: float
    query_mean: float
    rd_mean: float
    rm_mean: float
    ata_mean: float
    latency_ms_mean: float
    memory_mb_mean: float


def aggregate_samples(samples: Iterable[BenchmarkSample]) -> BenchmarkAggregate:
    rows: List[BenchmarkSample] = list(samples)
    if not rows:
        raise ValueError("Cannot aggregate empty benchmark sample list")

    return BenchmarkAggregate(
        n=len(rows),
        success_rate=mean(1.0 if r.success else 0.0 for r in rows),
        l0_mean=mean(r.l0 for r in rows),
        l2_mean=mean(r.l2 for r in rows),
        query_mean=mean(r.query_count for r in rows),
        rd_mean=mean(r.rd for r in rows),
        rm_mean=mean(r.rm for r in rows),
        ata_mean=mean(r.ata for r in rows),
        latency_ms_mean=mean(r.latency_ms for r in rows),
        memory_mb_mean=mean(r.memory_mb for r in rows),
    )


def as_report_dict(aggregate: BenchmarkAggregate) -> Dict[str, float | int]:
    return asdict(aggregate)
