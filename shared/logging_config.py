"""
Shared structured logging configuration for the AI Safety Platform.

Usage in each service:
    from shared.logging_config import configure_logging, CorrelationIDMiddleware
    import structlog

    configure_logging("my-service")
    logger = structlog.get_logger("my-service")

    app = FastAPI(...)
    app.add_middleware(CorrelationIDMiddleware)

Log format:
  ENV=development  → ConsoleRenderer (human-readable, timestamped)
  ENV=production   → JSONRenderer    (one JSON object per line, machine-parseable)

Every log line emitted inside a request automatically carries:
  request_id  — X-Request-ID header value (or auto-generated UUID)
  method      — HTTP method
  path        — request path
"""
import logging
import os
import uuid
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_IS_PRODUCTION = os.getenv("ENV", "development").lower() == "production"

# Processors shared between structlog loggers (our code) and stdlib loggers
# (third-party: asyncpg, presidio, deepeval, etc.)
_SHARED_PROCESSORS = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def configure_logging(service_name: str, level: int = logging.INFO) -> None:
    """
    Wire up structlog + stdlib bridge.

    Third-party loggers (asyncpg, presidio, sentence_transformers…) route
    through the same ProcessorFormatter so their output is also structured
    and carries the same request_id as our own log lines.
    """
    renderer = (
        structlog.processors.JSONRenderer()
        if _IS_PRODUCTION
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=_SHARED_PROCESSORS + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = []   # remove any prior handlers (e.g. a previous basicConfig call)
    root.addHandler(handler)
    root.setLevel(level)

    # Silence high-volume third-party loggers that add noise without value
    for _noisy in ("httpx", "httpcore", "opentelemetry", "sentence_transformers.base.model"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    structlog.get_logger(service_name).info(
        "logging_configured",
        service=service_name,
        mode="production" if _IS_PRODUCTION else "development",
    )


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware that:
      1. Reads X-Request-ID from the inbound request header, or generates a UUID if absent.
      2. Binds request_id (+ method, path) to structlog's contextvars so every log
         statement executed during this request automatically carries them.
      3. Clears contextvars after the response so the next request starts clean.
      4. Echoes the request_id back in the X-Request-ID response header.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
