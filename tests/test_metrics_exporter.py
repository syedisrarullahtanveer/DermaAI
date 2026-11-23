import unittest
from backend.monitoring.metrics_exporter import MetricsExporter

class TestMetricsExporter(unittest.TestCase):
    def test_prometheus_formatting(self):
        exp = MetricsExporter()
        exp.record_inference("melanoma", 45.2)
        exp.record_inference("melanoma", 50.8)
        output = exp.generate_prometheus_metrics()
        self.assertIn('dermaai_inferences_total{class="melanoma"} 2', output)
        self.assertIn("dermaai_latency_ms 48.00", output)

if __name__ == '__main__':
    unittest.main()
