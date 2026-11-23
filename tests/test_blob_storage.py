import unittest
import tempfile
from backend.storage.blob_storage import SecureBlobStorage

class TestBlobStorage(unittest.TestCase):
    def test_store_and_retrieve(self):
        with tempfile.TemporaryDirectory() as td:
            storage = SecureBlobStorage(td)
            data = b"image_raw_bytes"
            h = storage.store_image(data)
            retrieved = storage.get_image(h)
            self.assertEqual(retrieved, data)

if __name__ == '__main__':
    unittest.main()
