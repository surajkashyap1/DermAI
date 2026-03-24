from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.schemas.contracts import Citation
from app.services.providers import GenerationRequest
from app.services.retrieval import RetrievalHit, RetrievalService


Intent = Literal["greeting", "dermatology_qa", "off_topic", "emergency"]


EMERGENCY_PATTERNS = (
    "trouble breathing",
    "cannot breathe",
    "swollen lips",
    "swollen tongue",
    "rapidly spreading rash",
    "blistering",
    "sepsis",
    "high fever",
    "anaphylaxis",
)

OFF_TOPIC_PATTERNS = (
    "weather",
    "bitcoin",
    "football",
    "movie",
    "stock market",
    "recipe",
    "capital of",
)

GREETING_PATTERNS = {"hi", "hello", "hey", "yo", "good morning", "good evening"}

QUERY_REWRITES = {
    "what is melanoma": "melanoma definition skin cancer warning signs dermatologist evaluation",
    "define melanoma": "melanoma definition skin cancer warning signs dermatologist evaluation",
    "what is a melanoma": "melanoma definition skin cancer warning signs dermatologist evaluation",
    "what is skin cancer": "skin cancer melanoma lesion warning signs dermatologist evaluation",
}

IMAGE_LABEL_HINTS = {
    "suspicious_irregular_pattern": "irregular pigmented lesion melanoma warning signs urgent review dermoscopy biopsy",
    "uniform_benign_like_pattern": "benign nevus versus melanoma clinical distinction monitoring follow-up",
    "indeterminate_pigmented_pattern": "pigmented lesion indeterminate melanoma differential dermatologist evaluation follow-up",
    "low_quality_capture": "poor image quality lesion photo repeat capture dermatology evaluation",
}


class WorkflowState(TypedDict, total=False):
    message: str
    mode: str
    session_history: str
    image_analysis: dict[str, Any] | None
    normalized_message: str
    intent: Intent
    rewritten_query: str
    effective_query: str
    image_context: str
    retrieved_hits: list[RetrievalHit]
    top_score: float
    confidence: Literal["low", "medium", "high"]
    answer_text: str
    citations: list[Citation]
    follow_ups: list[str]
    disclaimer: str
    trace: list[str]


class ChatWorkflow:
    def __init__(self, retrieval: RetrievalService, provider) -> None:
        self.retrieval = retrieval
        self.provider = provider
        self.graph = self._build()

    def _append_trace(self, state: WorkflowState, label: str) -> list[str]:
        return [*(state.get("trace") or []), label]

    def _normalize(self, message: str) -> str:
        return re.sub(r"\s+", " ", message.strip().lower())

    def classify_intent(self, state: WorkflowState) -> WorkflowState:
        normalized = self._normalize(state["message"])
        intent: Intent = "dermatology_qa"

        if normalized in GREETING_PATTERNS or (len(normalized.split()) <= 3 and normalized.startswith(tuple(GREETING_PATTERNS))):
            intent = "greeting"
        elif any(pattern in normalized for pattern in EMERGENCY_PATTERNS):
            intent = "emergency"
        elif any(pattern in normalized for pattern in OFF_TOPIC_PATTERNS):
            intent = "off_topic"
        elif state.get("mode") == "image_follow_up":
            intent = "dermatology_qa"

        return {
            "normalized_message": normalized,
            "intent": intent,
            "trace": self._append_trace(state, f"classify:{intent}"),
        }

    def route_intent(self, state: WorkflowState) -> str:
        return state["intent"]

    def respond_greeting(self, state: WorkflowState) -> WorkflowState:
        image_hint = ""
        if state.get("image_analysis"):
            image_hint = " You can also ask what the uploaded image result means or what follow-up is reasonable."
        return {
            "answer_text": (
                "Hello. Ask me a dermatology question about lesions, melanoma warning signs, triage, "
                f"or low-confidence follow-up and I will answer with grounded evidence.{image_hint}"
            ),
            "confidence": "high",
            "citations": [],
            "follow_ups": [
                "What are warning signs of melanoma?",
                "How should low-confidence dermatology outputs be handled?",
            ],
            "disclaimer": "DermAI is informational only and does not replace a licensed dermatologist.",
            "trace": self._append_trace(state, "respond:greeting"),
        }

    def respond_off_topic(self, state: WorkflowState) -> WorkflowState:
        return {
            "answer_text": (
                "I can help with dermatology-focused questions, lesion warning signs, and cautious triage guidance. "
                "I am not meant for general-purpose chat."
            ),
            "confidence": "high",
            "citations": [],
            "follow_ups": [
                "What is melanoma?",
                "When should a lesion not be reassured as benign?",
            ],
            "disclaimer": "DermAI is informational only and does not replace a licensed dermatologist.",
            "trace": self._append_trace(state, "respond:off_topic"),
        }

    def respond_emergency(self, state: WorkflowState) -> WorkflowState:
        return {
            "answer_text": (
                "This sounds like a potentially urgent medical situation. DermAI should not handle possible "
                "emergencies such as breathing difficulty, severe blistering, or rapidly worsening systemic symptoms. "
                "Please seek urgent medical care immediately."
            ),
            "confidence": "high",
            "citations": [],
            "follow_ups": [
                "Do you want a summary of why this should be escalated urgently?",
            ],
            "disclaimer": "DermAI is informational only and does not replace emergency or specialist care.",
            "trace": self._append_trace(state, "respond:emergency"),
        }

    def rewrite_query(self, state: WorkflowState) -> WorkflowState:
        normalized = state["normalized_message"]
        rewritten = QUERY_REWRITES.get(normalized, normalized)

        if rewritten == normalized:
            tokens = normalized.split()
            if "melanoma" in tokens and "definition" not in tokens and "what" in tokens:
                rewritten = f"{normalized} definition skin cancer warning signs"
            elif len(tokens) <= 4:
                rewritten = f"{normalized} dermatology lesion triage"

        image_context = ""
        effective_query = rewritten
        image_analysis = state.get("image_analysis")
        if state.get("mode") == "image_follow_up" and image_analysis:
            predicted_class = image_analysis.get("predictedClass", "")
            summary = image_analysis.get("summary", "")
            quality = image_analysis.get("quality", {}) or {}
            issues = quality.get("issues", [])
            quality_note = f" Quality issues: {'; '.join(issues)}." if issues else ""
            image_context = (
                f"Uploaded image result: {predicted_class}. "
                f"Summary: {summary}. "
                f"Confidence band: {image_analysis.get('confidenceBand', 'unknown')}.{quality_note}"
            ).strip()
            hint = IMAGE_LABEL_HINTS.get(predicted_class, "lesion follow-up dermatology evaluation")
            effective_query = f"{rewritten} {hint}"

        return {
            "rewritten_query": rewritten,
            "effective_query": effective_query,
            "image_context": image_context,
            "trace": self._append_trace(state, "rewrite"),
        }

    def retrieve(self, state: WorkflowState) -> WorkflowState:
        hits = self.retrieval.search(state.get("effective_query") or state["rewritten_query"], top_k=4)
        top_score = hits[0].score if hits else 0.0

        if len(hits) >= 3 and top_score >= 1.0:
            confidence: Literal["low", "medium", "high"] = "high"
        elif len(hits) >= 2 and top_score >= 0.45:
            confidence = "medium"
        else:
            confidence = "low"

        image_analysis = state.get("image_analysis") or {}
        if state.get("mode") == "image_follow_up":
            image_band = image_analysis.get("confidenceBand")
            if image_band == "low":
                confidence = "low" if confidence == "medium" else "medium" if confidence == "high" else "low"
            elif image_band == "medium" and confidence == "high":
                confidence = "medium"

        return {
            "retrieved_hits": hits,
            "top_score": top_score,
            "confidence": confidence,
            "trace": self._append_trace(state, f"retrieve:{len(hits)}"),
        }

    def route_retrieval(self, state: WorkflowState) -> str:
        return "no_hits" if not state.get("retrieved_hits") else "generate"

    def respond_no_hits(self, state: WorkflowState) -> WorkflowState:
        return {
            "answer_text": (
                "I do not have enough grounded dermatology evidence in the current corpus to answer that well yet. "
                "Try a more specific skin-lesion or triage question."
            ),
            "confidence": "low",
            "citations": [],
            "follow_ups": [
                "What are warning signs of melanoma?",
                "How should low-confidence lesion outputs be handled?",
            ],
            "disclaimer": "DermAI is informational only and does not replace a licensed dermatologist.",
            "trace": self._append_trace(state, "respond:no_hits"),
        }

    async def generate_answer(self, state: WorkflowState) -> WorkflowState:
        hits = state["retrieved_hits"]
        context = "\n\n".join(
            f"[{index}] {hit.title} | {hit.section} | {hit.text}"
            for index, hit in enumerate(hits, start=1)
        )
        generation = await self.provider.generate(
            GenerationRequest(
                user_message=state["message"],
                retrieved_context=context,
                confidence=state["confidence"],
                intent=state["intent"],
                mode=state.get("mode", "chat"),
                conversation_history=state.get("session_history", ""),
                image_context=state.get("image_context", ""),
            )
        )
        return {
            "answer_text": generation.answer,
            "trace": self._append_trace(state, f"generate:{generation.provider}"),
        }

    def format_response(self, state: WorkflowState) -> WorkflowState:
        hits = state.get("retrieved_hits") or []
        citations = [
            Citation(
                id=hit.id,
                title=f"{hit.title} - {hit.section}",
                source=f"{hit.source} ({hit.year})",
                snippet=hit.snippet,
                href=hit.href,
            )
            for hit in hits
        ]

        if state["intent"] == "dermatology_qa" and state["confidence"] == "low":
            follow_ups = [
                "Would you like me to narrow this to warning signs, triage, or follow-up?",
                "Do you want the supporting evidence snippets shown first?",
            ]
        elif state.get("mode") == "image_follow_up":
            follow_ups = [
                "What features usually justify urgent in-person review?",
                "What should be monitored if the image result stays uncertain?",
            ]
        elif state["intent"] == "dermatology_qa":
            follow_ups = [
                "Do you want a short checklist version of this answer?",
                "Would you like the key supporting evidence summarized?",
            ]
        else:
            follow_ups = state.get("follow_ups") or []

        return {
            "citations": citations if hits else state.get("citations", []),
            "follow_ups": follow_ups,
            "disclaimer": state.get(
                "disclaimer",
                "DermAI is informational only and does not replace a licensed dermatologist.",
            ),
            "trace": self._append_trace(state, "format"),
        }

    def _build(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("respond_greeting", self.respond_greeting)
        graph.add_node("respond_off_topic", self.respond_off_topic)
        graph.add_node("respond_emergency", self.respond_emergency)
        graph.add_node("rewrite_query", self.rewrite_query)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("respond_no_hits", self.respond_no_hits)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("format_response", self.format_response)

        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self.route_intent,
            {
                "greeting": "respond_greeting",
                "off_topic": "respond_off_topic",
                "emergency": "respond_emergency",
                "dermatology_qa": "rewrite_query",
            },
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self.route_retrieval,
            {
                "no_hits": "respond_no_hits",
                "generate": "generate_answer",
            },
        )
        graph.add_edge("generate_answer", "format_response")
        graph.add_edge("respond_greeting", END)
        graph.add_edge("respond_off_topic", END)
        graph.add_edge("respond_emergency", END)
        graph.add_edge("respond_no_hits", END)
        graph.add_edge("format_response", END)
        return graph.compile()
