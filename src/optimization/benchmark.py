"""
Latency and throughput benchmark harness for DermaAI inference pipelines.
"""
import time
import numpy as np

class LatencyBenchmarkHarness:
    def __init__(self, warmup_runs: int = 5, test_runs: int = 20):
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs

    def benchmark(self, infer_fn, sample_input: np.ndarray) -> dict:
        for _ in range(self.warmup_runs):
            _ = infer_fn(sample_input)
            
        latencies = []
        for _ in range(self.test_runs):
            t0 = time.perf_counter()
            _ = infer_fn(sample_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            
        return {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "throughput_fps": float(1000.0 / np.mean(latencies))
        }
