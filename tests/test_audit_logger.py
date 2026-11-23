import unittest
import tempfile
import os
from backend.logging.audit_logger import AuditLogger

class TestAuditLogger(unittest.TestCase):
    def test_event_logging(self):
        with tempfile.TemporaryDirectory() as td:
            log_file = os.path.join(td, "audit.log")
            logger = AuditLogger(log_file)
            logger.log_prediction_event("REQ-1", "127.0.0.1", "v2.0", "Melanoma", 0.94)
            self.assertTrue(os.path.exists(log_file))
            with open(log_file) as f:
                content = f.read()
                self.assertIn("Melanoma", content)

if __name__ == '__main__':
    unittest.main()
