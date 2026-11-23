import unittest
from backend.batch.batch_processor import BatchPredictionProcessor

class TestBatchProcessor(unittest.TestCase):
    def test_batching(self):
        bp = BatchPredictionProcessor(max_batch_size=2)
        bp.enqueue("1", "data1")
        bp.enqueue("2", "data2")
        bp.enqueue("3", "data3")
        batch1 = bp.dequeue_batch()
        self.assertEqual(len(batch1), 2)
        batch2 = bp.dequeue_batch()
        self.assertEqual(len(batch2), 1)

if __name__ == '__main__':
    unittest.main()
