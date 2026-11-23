import unittest
import tempfile
import os
from src.evaluation.metric_snapshot import MetricSnapshotLogger

class TestMetricSnapshot(unittest.TestCase):
    def test_record_and_export(self):
        logger = MetricSnapshotLogger()
        logger.record_epoch(1, {"loss": 0.45, "val_acc": 0.88})
        logger.record_epoch(2, {"loss": 0.38, "val_acc": 0.91})
        self.assertEqual(len(logger.history), 2)
        
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "log.csv")
            json_path = os.path.join(td, "log.json")
            logger.export_csv(csv_path)
            logger.export_json(json_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))

if __name__ == '__main__':
    unittest.main()
