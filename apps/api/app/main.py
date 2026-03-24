import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.images import router as images_router
from app.api.routes.system import router as system_router
from app.core.config import settings
from app.services.chat_runtime import chat_runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dermai.api")

app = FastAPI(
    title="DermAI API",
    version=settings.version,
    summary="DermAI API with grounded dermatology chat, demo vision analysis, and multimodal follow-up.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def log_runtime_configuration() -> None:
    retrieval_status = chat_runtime.retrieval.status()
    logger.info(
        "Retrieval backend=%s qdrant_url=%s collection=%s reranker_enabled=%s corpus_documents=%s corpus_chunks=%s",
        retrieval_status["backend"],
        retrieval_status["qdrant_url"] or "embedded-local-store",
        retrieval_status["qdrant_collection"],
        retrieval_status["reranker_enabled"],
        retrieval_status["corpus_documents"],
        retrieval_status["corpus_chunks"],
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "%s %s -> 500 (%sms) request_id=%s",
            request.method,
            request.url.path,
            duration_ms,
            request_id,
        )
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "DermAI hit an unexpected server error.",
                "requestId": request_id,
            },
        )
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "%s %s -> %s (%sms) request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

    response.headers["x-request-id"] = request_id
    return response


app.include_router(system_router)
app.include_router(chat_router)
app.include_router(images_router)
