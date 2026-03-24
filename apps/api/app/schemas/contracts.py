from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    environment: str


class VersionResponse(BaseModel):
    name: str
    version: str
    commitSha: str
    apiBasePath: str


class Citation(BaseModel):
    id: str
    title: str
    source: str
    snippet: str
    href: str | None = None


class ChatRequest(BaseModel):
    sessionId: str | None = None
    message: str
    mode: Literal["chat", "image_follow_up"]


class ChatResponse(BaseModel):
    sessionId: str
    answer: str
    citations: list[Citation]
    confidence: Literal["low", "medium", "high"]
    disclaimer: str
    followUps: list[str]


class VisionPrediction(BaseModel):
    label: str
    confidence: float
    rationale: str


class VisionQuality(BaseModel):
    usable: bool
    issues: list[str]
    contrast: float
    sharpness: float
    lesionCoverage: float
    asymmetry: float


class ImageAnalysis(BaseModel):
    predictedClass: str
    confidence: float
    confidenceBand: Literal["low", "medium", "high"]
    summary: str
    caution: str
    topPredictions: list[VisionPrediction]
    quality: VisionQuality
    overlayImageDataUrl: str
    width: int
    height: int


class UploadImageResponse(BaseModel):
    sessionId: str
    status: Literal["pending", "completed"]
    message: str
    imageAnalysis: ImageAnalysis | None = None


class SessionMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
