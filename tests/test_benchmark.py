import unittest
import time
import numpy as np
from src.optimization.benchmark import LatencyBenchmarkHarness

class TestBenchmarkHarness(unittest.TestCase):
    def test_harness_metrics(self):
        harness = LatencyBenchmarkHarness(warmup_runs=2, test_runs=5)
        res = harness.benchmark(lambda x: time.sleep(0.001), np.ones((2, 2)))
        self.assertIn("mean_ms", res)
        self.assertIn("p95_ms", res)
        self.assertGreater(res["throughput_fps"], 0.0)

if __name__ == '__main__':
    unittest.main()
