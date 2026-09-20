"""DermAI FastAPI backend.

A thin HTTP layer over the ``dermai`` engine (7-class HAM10000 CNN + Grad-CAM and
the LangChain + FAISS RAG pipeline on Groq). Designed to run on Render; the
Next.js web app on Vercel calls these endpoints.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from pathlib import Path

# Make the repo-root ``dermai`` package importable regardless of the working dir.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from dermai import __version__
from dermai.classifier import CLASSES, SkinLesionClassifier
from dermai.config import groq_enabled
from dermai.rag import DermatologyRAG
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ClassificationResponse,
    ClassProbability,
    HealthResponse,
    SourceRef,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dermai.api")

app = FastAPI(title="DermAI API", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily-loaded singletons (models/index load on first use, not at import).
_classifier = SkinLesionClassifier()
_rag = DermatologyRAG()


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        modelAvailable=_classifier.is_available,
        llmBackend="groq" if groq_enabled() else "extractive-fallback",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    history = [(turn.user, turn.assistant) for turn in payload.history]
    result = _rag.ask(message, chat_history=history)
    return ChatResponse(
        answer=result.answer,
        provider=result.provider,
        sources=[
            SourceRef(source=s["source"], page=s.get("page"), snippet=s["snippet"])
            for s in result.sources
        ],
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify(file: UploadFile = File(...)) -> ClassificationResponse:
    if not _classifier.is_available:
        raise HTTPException(
            status_code=503,
            detail="Classifier weights are not available on the server.",
        )
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")

    payload = await file.read()
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Could not read the image.") from error

    try:
        prediction = _classifier.predict(image, with_gradcam=True)
    except Exception as error:  # noqa: BLE001
        logger.exception("classification failed")
        raise HTTPException(status_code=500, detail="Image analysis failed.") from error

    info = prediction.class_info
    overlay_url = _image_to_data_url(prediction.overlay) if prediction.overlay else None
    return ClassificationResponse(
        predictedClass=info.name,
        code=info.code,
        malignancy=info.malignancy,
        description=info.description,
        confidence=round(prediction.confidence, 4),
        gradcamImageDataUrl=overlay_url,
        probabilities=[
            ClassProbability(
                code=c.code,
                name=c.name,
                probability=round(prediction.probabilities[c.index], 4),
            )
            for c in CLASSES
        ],
    )
