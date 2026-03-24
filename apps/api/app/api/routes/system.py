from fastapi import APIRouter

from app.core.config import settings
from app.schemas.contracts import HealthResponse, VersionResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        service=settings.service_name,
        environment=settings.env,
    )


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    return VersionResponse(
        name="DermAI API",
        version=settings.version,
        commitSha=settings.commit_sha,
        apiBasePath="/",
    )
