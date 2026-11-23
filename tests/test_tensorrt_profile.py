import unittest
from src.optimization.tensorrt_profile import TensorRTProfiler

class TestTensorRTProfiler(unittest.TestCase):
    def test_acceleration_calc(self):
        prof = TensorRTProfiler()
        res = prof.estimate_acceleration(50.0, "fp16")
        self.assertLess(res["optimized_ms"], 50.0)

if __name__ == '__main__':
    unittest.main()
