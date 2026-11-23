import unittest, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.batch_helpers import helper_13

class TestHelper13(unittest.TestCase):
    def test_add(self):
        self.assertEqual(helper_13(10), 10 + 13)

if __name__ == '__main__':
    unittest.main()
