"""OWASP Clinical Security and Content Security Policy (CSP) Headers Middleware."""

try:
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    class BaseHTTPMiddleware:
        def __init__(self, app):
            self.app = app


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces clinical-grade HTTP security headers across all API responses."""

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "frame-ancestors 'none';"
        ),
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value
        return response
