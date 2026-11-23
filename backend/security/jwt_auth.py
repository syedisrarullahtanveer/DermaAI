"""
Lightweight cryptographic JWT authentication handler for clinical practitioner access.
"""
import base64
import hashlib
import hmac
import json
import time

class JWTAuthHandler:
    def __init__(self, secret: str = "derma_secret_key_2026"):
        self.secret = secret.encode("utf-8")

    def _b64encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    def _b64decode(self, data: str) -> bytes:
        padding = 4 - (len(data) % 4)
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data.encode("utf-8"))

    def create_token(self, user_id: str, role: str = "clinician", expires_in: int = 3600) -> str:
        header = json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8")
        payload = json.dumps({
            "sub": user_id,
            "role": role,
            "exp": int(time.time()) + expires_in
        }).encode("utf-8")
        seg1 = self._b64encode(header)
        seg2 = self._b64encode(payload)
        sig = hmac.new(self.secret, f"{seg1}.{seg2}".encode("utf-8"), hashlib.sha256).digest()
        seg3 = self._b64encode(sig)
        return f"{seg1}.{seg2}.{seg3}"

    def verify_token(self, token: str) -> dict | None:
        parts = token.split(".")
        if len(parts) != 3: return None
        seg1, seg2, seg3 = parts
        sig = hmac.new(self.secret, f"{seg1}.{seg2}".encode("utf-8"), hashlib.sha256).digest()
        if not hmac.compare_digest(sig, self._b64decode(seg3)):
            return None
        payload = json.loads(self._b64decode(seg2))
        if payload.get("exp", 0) < time.time():
            return None # Expired
        return payload
