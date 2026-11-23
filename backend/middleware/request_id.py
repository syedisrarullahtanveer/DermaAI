import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIdMiddleware(BaseHTTPMiddleware):
    """Add a unique X-Request-ID header to each request.

    The request ID is generated with ``uuid.uuid4()`` and stored
    on the request state for downstream handlers.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers['X-Request-ID'] = request_id
        return response
