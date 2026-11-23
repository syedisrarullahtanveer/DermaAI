import unittest, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.batch_helpers import helper_5

class TestHelper5(unittest.TestCase):
    def test_add(self):
        self.assertEqual(helper_5(10), 10 + 5)

if __name__ == '__main__':
    unittest.main()
