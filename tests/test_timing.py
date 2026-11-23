import unittest
import io
import sys
from contextlib import redirect_stdout
from src.utils.engineering_utils import timing

@timing
def add(a, b):
    return a + b

class TestTiming(unittest.TestCase):
    def test_output(self):
        f = io.StringIO()
        with redirect_stdout(f):
            result = add(2, 3)
        self.assertEqual(result, 5)
        output = f.getvalue()
        self.assertIn('[TIMING] add took', output)

if __name__ == '__main__':
    unittest.main()
