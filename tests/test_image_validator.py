"""Unit tests for ImagePayloadValidator."""

import unittest
from backend.security.image_validator import ImagePayloadValidator


class TestImagePayloadValidator(unittest.TestCase):
    def setUp(self):
        self.validator = ImagePayloadValidator(max_size_bytes=1024)

    def test_valid_jpeg(self):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        is_valid, fmt, err = self.validator.validate(jpeg_bytes)
        self.assertTrue(is_valid)
        self.assertEqual(fmt, "jpeg")
        self.assertIsNone(err)

    def test_valid_png(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        is_valid, fmt, err = self.validator.validate(png_bytes)
        self.assertTrue(is_valid)
        self.assertEqual(fmt, "png")
        self.assertIsNone(err)

    def test_rejects_php_script(self):
        php_bytes = b"<?php echo 'malicious'; ?>"
        is_valid, fmt, err = self.validator.validate(php_bytes)
        self.assertFalse(is_valid)
        self.assertIsNone(fmt)
        self.assertIn("Unsupported or spoofed", err)

    def test_rejects_oversized_payload(self):
        huge_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 2000
        is_valid, fmt, err = self.validator.validate(huge_bytes)
        self.assertFalse(is_valid)
        self.assertIn("exceeds limit", err)


if __name__ == "__main__":
    unittest.main()
