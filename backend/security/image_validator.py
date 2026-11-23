"""Dermoscopic Image Payload Security and Magic Byte Signature Validator."""

from typing import Tuple, Optional


class ImagePayloadValidator:
    """Validate image binary streams against known magic byte signatures."""

    MAGIC_SIGNATURES = {
        "jpeg": [
            b"\xff\xd8\xff\xe0",  # JFIF
            b"\xff\xd8\xff\xe1",  # EXIF
            b"\xff\xd8\xff\xee",
            b"\xff\xd8\xff\xdb",
        ],
        "png": [b"\x89PNG\r\n\x1a\n"],
    }

    def __init__(self, max_size_bytes: int = 15 * 1024 * 1024) -> None:
        """Initialize validator.

        Args:
            max_size_bytes: Maximum allowed payload size in bytes (default 15 MB).
        """
        self.max_size_bytes = max_size_bytes

    def validate(self, payload: bytes) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate payload size and header signature.

        Args:
            payload: Raw bytes of the uploaded file.

        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (is_valid, detected_format, error_message).
        """
        if not payload:
            return False, None, "Payload is empty"

        if len(payload) > self.max_size_bytes:
            return False, None, f"Payload size ({len(payload)} bytes) exceeds limit ({self.max_size_bytes} bytes)"

        # Check PNG
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return True, "png", None

        # Check JPEG
        if payload.startswith(b"\xff\xd8\xff"):
            return True, "jpeg", None

        return False, None, "Unsupported or spoofed image file header signature"
