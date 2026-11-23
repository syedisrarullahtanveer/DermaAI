"""
Local filesystem and mock Cloud Storage blob manager with SHA256 content deduplication.
"""
import hashlib
import os

class SecureBlobStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def store_image(self, image_bytes: bytes) -> str:
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        filename = f"{content_hash}.png"
        full_path = os.path.join(self.base_path, filename)
        if not os.path.exists(full_path):
            with open(full_path, "wb") as f:
                f.write(image_bytes)
        return content_hash

    def get_image(self, content_hash: str) -> bytes | None:
        filename = f"{content_hash}.png"
        full_path = os.path.join(self.base_path, filename)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                return f.read()
        return None
