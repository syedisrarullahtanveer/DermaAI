import unittest
from src.utils.engineering_utils import memoize

counter = 0

@memoize
def inc(x):
    global counter
    counter += x
    return counter

class TestMemoize(unittest.TestCase):
    def test_caching(self):
        global counter
        counter = 0
        self.assertEqual(inc(1), 1)   # first call, computes
        self.assertEqual(counter, 1)
        self.assertEqual(inc(1), 1)   # cached, counter unchanged
        self.assertEqual(counter, 1)
        self.assertEqual(inc(2), 3)   # new args, computes
        self.assertEqual(counter, 3)

if __name__ == "__main__":
    unittest.main()
