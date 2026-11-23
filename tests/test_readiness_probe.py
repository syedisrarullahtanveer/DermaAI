import unittest
from backend.health.readiness_probe import ReadinessProbe

class TestReadinessProbe(unittest.TestCase):
    def test_probe_lifecycle(self):
        probe = ReadinessProbe()
        self.assertEqual(probe.check_health()["status"], "warming_up")
        probe.mark_ready("sha256_mock_hash")
        self.assertEqual(probe.check_health()["status"], "ready")

if __name__ == '__main__':
    unittest.main()
