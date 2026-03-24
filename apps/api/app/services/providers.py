from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from app.core.config import settings


@dataclass
class GenerationRequest:
    user_message: str
    retrieved_context: str
    confidence: str
    intent: str
    mode: str
    conversation_history: str = ""
    image_context: str = ""
    evidence_summary: str = ""


@dataclass
class GenerationResult:
    answer: str
    provider: str
    model: str


class ChatProvider(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        ...


class ExtractiveFallbackProvider:
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        evidence_lines = [line.strip() for line in request.retrieved_context.splitlines() if line.strip()]
        summary = " ".join(evidence_lines[:4])
        image_prefix = f"Image context: {request.image_context}. " if request.image_context else ""
        answer = (
            f"{image_prefix}Here is a direct summary based on the available dermatology material: {summary}"
        )
        return GenerationResult(answer=answer, provider="fallback", model="extractive-summary")


class GroqProvider:
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.groq_model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are DermAI, a dermatology assistant for normal users. "
                        "The app supports image upload, and users can ask follow-up questions about an uploaded image. "
                        "Use only the supplied evidence. "
                        "Write in clear, plain language that a non-expert can understand. "
                        "Do not sound robotic, defensive, or repetitive. "
                        "Do not give definitive diagnosis language. "
                        "If the user asks for a definition, start with a direct one-sentence definition. "
                        "If image context is supplied, treat it as non-diagnostic supporting context only. "
                        "Do not keep repeating advice to see a dermatologist unless the situation is clearly urgent. "
                        "Do not mention confidence levels, evidence coverage, or internal system wording unless the user explicitly asks. "
                        "If the user asks what the app can do, mention both chat and image upload. "
                        "Keep the answer concise, direct, and natural."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {request.user_message}\n\n"
                        f"Mode: {request.mode}\n"
                        f"Conversation History:\n{request.conversation_history or 'None'}\n\n"
                        f"Image Context:\n{request.image_context or 'None'}\n\n"
                        f"Evidence Coverage Summary:\n{request.evidence_summary or 'None'}\n\n"
                        f"Evidence:\n{request.retrieved_context}\n\n"
                        f"Confidence: {request.confidence}\n"
                        f"Intent: {request.intent}\n\n"
                        "Write a concise grounded answer in two short paragraphs or fewer. "
                        "Answer the covered parts directly and in plain English. "
                        "If image context is present, explain how it may relate to the retrieved evidence without treating it as confirmed diagnosis. "
                        "If the question is broad and the material only covers part of it, answer the part that is clearly supported without using phrases like low-confidence or evidence coverage. "
                        "Do not add warnings or disclaimers unless the prompt is clearly about urgent danger signs. "
                        "Do not mention information that is not in the evidence."
                    ),
                },
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.groq_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"]["content"].strip()
            return GenerationResult(answer=content, provider="groq", model=settings.groq_model)
        except Exception:
            fallback = ExtractiveFallbackProvider()
            return await fallback.generate(request)


def get_chat_provider() -> ChatProvider:
    if settings.groq_api_key:
        return GroqProvider()
    return ExtractiveFallbackProvider()
