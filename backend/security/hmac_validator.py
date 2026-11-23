"""
HMAC request signature verification for tamper-proof API client authentication.
"""
import hashlib
import hmac

class HMACSignatureValidator:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode("utf-8")

    def generate_signature(self, payload: bytes) -> str:
        return hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()

    def verify_signature(self, payload: bytes, provided_signature: str) -> bool:
        expected = self.generate_signature(payload)
        return hmac.compare_digest(expected, provided_signature)
