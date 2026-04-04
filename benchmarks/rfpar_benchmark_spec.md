# RFPAR Benchmark Spec (Scaffold)

## Purpose
Track parity against paper-reported trends (Tables 1-4 in arXiv:2502.07821).

## Required Metrics
- Success rate (classification)
- L0 and L2 perturbation norms
- Query count
- RD (mAP reduction)
- RM (object removal rate)
- ATA (attacked area)
- Latency (ms)
- Memory (MB)

## Evaluation Buckets
1. ImageNet-1K (classification): alpha=0.01, T=3
2. MS-COCO (detection): alpha in [0.01,0.05], T=20
3. Argoverse (detection): alpha=0.05, large-resolution stress case

## Reporting
- Per-run JSON summary in `outputs/<run>/benchmark.json`
- Aggregate report in `reports/BENCHMARK_REPORT.md`
