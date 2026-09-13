from __future__ import annotations

import time
from collections import defaultdict

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

RATE_LIMIT = 100
RATE_WINDOW = 60


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if self._api_key is not None:
            provided = request.headers.get("X-API-Key")
            if provided != self._api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "message": "Invalid or missing API key"},
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._counters: dict[str, list] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        hits = self._counters[client_ip]
        cutoff = now - RATE_WINDOW
        self._counters[client_ip] = [t for t in hits if t > cutoff]
        hits = self._counters[client_ip]

        if len(hits) >= RATE_LIMIT:
            oldest = min(hits) if hits else now
            retry_after = int(RATE_WINDOW - (now - oldest)) + 1
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "message": "Too many requests"},
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        hits.append(now)
        return await call_next(request)


def add_middleware(app: FastAPI, api_key: str | None = None) -> None:
    app.add_middleware(RateLimitMiddleware)
    if api_key is not None:
        app.add_middleware(AuthMiddleware, api_key=api_key)
