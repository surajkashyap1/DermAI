import logging

from fastapi import APIRouter, Request

from app.schemas.contracts import ChatRequest, ChatResponse, SessionResponse
from app.services.chat_runtime import chat_runtime

router = APIRouter(tags=["chat"])
logger = logging.getLogger("dermai.api.chat")


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    logger.info(
        "chat_request session_id=%s mode=%s request_id=%s",
        payload.sessionId,
        payload.mode,
        getattr(request.state, "request_id", "unknown"),
    )
    return await chat_runtime.answer(payload.sessionId, payload.message, payload.mode)


@router.get("/session/{session_id}", response_model=SessionResponse)
async def session(session_id: str) -> SessionResponse:
    return chat_runtime.sessions.to_response(session_id)
