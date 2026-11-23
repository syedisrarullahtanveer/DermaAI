import unittest
from backend.streaming.sse_progress import SSEProgressGenerator

class TestSSEProgress(unittest.TestCase):
    def test_sse_formatting(self):
        gen = SSEProgressGenerator()
        event = gen.format_event("hair_removal", 25, "Filtering artifacts")
        self.assertTrue(event.startswith("event: progress"))
        self.assertIn("hair_removal", event)

if __name__ == '__main__':
    unittest.main()
