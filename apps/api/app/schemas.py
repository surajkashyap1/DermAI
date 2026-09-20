"""Pydantic request/response models for the DermAI API."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    modelAvailable: bool
    llmBackend: str


class ChatTurn(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


class SourceRef(BaseModel):
    source: str
    page: int | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    provider: str
    sources: list[SourceRef] = []


class ClassProbability(BaseModel):
    code: str
    name: str
    probability: float


class ClassificationResponse(BaseModel):
    predictedClass: str
    code: str
    malignancy: str
    description: str
    confidence: float
    gradcamImageDataUrl: str | None = None
    probabilities: list[ClassProbability]
