"""
In-memory batch aggregator for parallel client inference requests.
"""
class BatchPredictionProcessor:
    def __init__(self, max_batch_size: int = 16):
        self.max_batch_size = max_batch_size
        self.queue = []

    def enqueue(self, item_id: str, item_data):
        self.queue.append((item_id, item_data))

    def dequeue_batch(self) -> list:
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        return batch
