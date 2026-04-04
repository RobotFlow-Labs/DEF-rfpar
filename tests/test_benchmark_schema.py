from anima_rfpar.benchmarks import BenchmarkSample, aggregate_samples


def test_aggregate_samples() -> None:
    rows = [
        BenchmarkSample(success=True, l0=1.0, l2=0.5, query_count=10),
        BenchmarkSample(success=False, l0=2.0, l2=1.5, query_count=20),
    ]
    agg = aggregate_samples(rows)
    assert agg.n == 2
    assert agg.query_mean == 15
