from fastapi import APIRouter, HTTPException

from app.schemas.contracts import Citation
from app.services.chat_runtime import chat_runtime

router = APIRouter(tags=["citations"])


@router.get("/citations/{citation_id}", response_model=Citation)
async def get_citation(citation_id: str) -> Citation:
    hit = chat_runtime.retrieval.citation_by_id(citation_id)
    if hit:
        return Citation(
            id=hit.id,
            title=f"{hit.title} - {hit.section}",
            source=f"{hit.source} ({hit.year})",
            snippet=hit.text,
            href=hit.href,
        )

    raise HTTPException(status_code=404, detail=f"Citation '{citation_id}' was not found.")
