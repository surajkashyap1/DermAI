from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.schemas.contracts import ChatResponse, ImageAnalysis, SessionMessage, SessionResponse
from app.services.chat_workflow import ChatWorkflow
from app.services.providers import get_chat_provider
from app.services.retrieval import RetrievalService


@dataclass
class SessionState:
    session_id: str
    messages: list[SessionMessage]
    image_analysis_available: bool = False
    image_analysis: ImageAnalysis | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str | None) -> SessionState:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        next_id = session_id or str(uuid4())
        state = SessionState(session_id=next_id, messages=[])
        self._sessions[next_id] = state
        return state

    def add_message(self, session_id: str, message: SessionMessage) -> None:
        self._sessions[session_id].messages.append(message)

    def to_response(self, session_id: str) -> SessionResponse:
        state = self._sessions.get(session_id)
        if not state:
            return SessionResponse(
                sessionId=session_id,
                messages=[],
                imageAnalysisAvailable=False,
                imageAnalysis=None,
            )

        return SessionResponse(
            sessionId=state.session_id,
            messages=state.messages,
            imageAnalysisAvailable=state.image_analysis_available,
            imageAnalysis=state.image_analysis,
        )

    def attach_image_analysis(self, session_id: str, image_analysis: ImageAnalysis) -> None:
        state = self.get_or_create(session_id)
        state.image_analysis_available = True
        state.image_analysis = image_analysis
        self.add_message(
            session_id,
            SessionMessage(
                id=str(uuid4()),
                role="assistant",
                content=(
                    "Image analysis added to this session. "
                    f"Predicted visual pattern: {image_analysis.predictedClass}. "
                    f"Summary: {image_analysis.summary}"
                ),
            ),
        )


class ChatRuntime:
    def __init__(self) -> None:
        self.retrieval = RetrievalService()
        self.provider = get_chat_provider()
        self.workflow = ChatWorkflow(self.retrieval, self.provider)
        self.sessions = SessionStore()

    async def answer(self, session_id: str | None, message: str, mode: str = "chat") -> ChatResponse:
        session = self.sessions.get_or_create(session_id)
        user_message = SessionMessage(id=str(uuid4()), role="user", content=message)
        self.sessions.add_message(session.session_id, user_message)
        recent_history = session.messages[-6:]
        history_text = "\n".join(f"{item.role}: {item.content}" for item in recent_history)

        state = await self.workflow.graph.ainvoke(
            {
                "message": message,
                "mode": mode,
                "session_history": history_text,
                "image_analysis": session.image_analysis.model_dump() if session.image_analysis else None,
                "trace": [],
            }
        )

        assistant_message = SessionMessage(
            id=str(uuid4()),
            role="assistant",
            content=state["answer_text"],
        )
        self.sessions.add_message(session.session_id, assistant_message)

        return ChatResponse(
            sessionId=session.session_id,
            answer=state["answer_text"],
            citations=state.get("citations", []),
            confidence=state["confidence"],
            disclaimer=state["disclaimer"],
            followUps=state.get("follow_ups", []),
        )


chat_runtime = ChatRuntime()
