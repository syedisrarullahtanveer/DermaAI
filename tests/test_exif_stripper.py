import unittest
import io
from PIL import Image
from backend.security.exif_stripper import EXIFMetadataStripper

class TestEXIFStripper(unittest.TestCase):
    def test_stripping(self):
        stripper = EXIFMetadataStripper()
        img = Image.new("RGB", (20, 20), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        cleaned = stripper.strip_metadata(raw)
        self.assertGreater(len(cleaned), 0)

if __name__ == '__main__':
    unittest.main()
