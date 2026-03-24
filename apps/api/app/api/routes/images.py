import logging

from fastapi import APIRouter, File, Request, UploadFile
from fastapi import HTTPException

from app.schemas.contracts import UploadImageResponse
from app.services.chat_runtime import chat_runtime
from app.services.vision import vision_service

router = APIRouter(tags=["vision"])
logger = logging.getLogger("dermai.api.vision")


@router.post("/upload-image", response_model=UploadImageResponse)
async def upload_image(request: Request, file: UploadFile = File(...)) -> UploadImageResponse:
    payload = await file.read()
    session = chat_runtime.sessions.get_or_create(None)
    logger.info(
        "upload_request session_id=%s filename=%s request_id=%s",
        session.session_id,
        file.filename,
        getattr(request.state, "request_id", "unknown"),
    )

    try:
        image_analysis = vision_service.analyze(payload, file.content_type)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="DermAI could not run image analysis right now.") from error

    chat_runtime.sessions.attach_image_analysis(session.session_id, image_analysis)

    return UploadImageResponse(
        sessionId=session.session_id,
        status="completed",
        message="Image analysis completed. You can now ask a follow-up question in the same session.",
        imageAnalysis=image_analysis,
    )
