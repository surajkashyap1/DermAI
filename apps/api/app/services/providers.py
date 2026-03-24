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
            "DermAI is using its local fallback answer mode because no hosted provider is configured. "
            f"{image_prefix}Based on the retrieved dermatology evidence, the key points are: "
            f"{summary}"
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
                        "You are DermAI, a cautious dermatology evidence assistant. "
                        "Use only the supplied evidence. If the evidence is limited, say so clearly. "
                        "Do not give definitive diagnosis language. "
                        "If the user asks for a definition, start with a direct one-sentence definition. "
                        "If image context is supplied, treat it as non-diagnostic supporting context only. "
                        "Keep the answer concise and product-grade."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {request.user_message}\n\n"
                        f"Mode: {request.mode}\n"
                        f"Conversation History:\n{request.conversation_history or 'None'}\n\n"
                        f"Image Context:\n{request.image_context or 'None'}\n\n"
                        f"Evidence:\n{request.retrieved_context}\n\n"
                        f"Confidence: {request.confidence}\n"
                        f"Intent: {request.intent}\n\n"
                        "Write a concise grounded answer in two short paragraphs or fewer. "
                        "If image context is present, explain how it may relate to the retrieved evidence without treating it as confirmed diagnosis. "
                        "Mention uncertainty when confidence is low. "
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
