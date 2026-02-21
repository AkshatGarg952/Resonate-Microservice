"""
Intervention suggestion routes.
"""
from fastapi import APIRouter, HTTPException, Depends, Request

from app.models.schemas import InterventionRequest
from app.services import openai_service
from app.core.logger import log_request, log_error
from app.core.auth import verify_internal_secret
from app.core.limiter import limiter

router = APIRouter(dependencies=[Depends(verify_internal_secret)])


@router.post("/generate-interventions")
@limiter.limit("10/minute")
async def generate_interventions(request: Request, req: InterventionRequest):
    """
    Suggest personalized health interventions based on memory context.

    Takes the structured memoryContext built by the server's MemoryContextBuilder
    and returns 3–5 AI-generated actionable intervention suggestions.
    """
    log_request("/generate-interventions")

    try:
        result = await openai_service.generate_interventions(
            memory_context=req.memoryContext,
            gender=req.gender,
            age=req.age,
        )
        return {"status": "success", "suggestions": result.get("suggestions", [])}
    except ValueError as e:
        log_error("Intervention generation", e)
        raise HTTPException(status_code=500, detail="AI generation failed to produce valid JSON")
    except Exception as e:
        log_error("Intervention generation", e)
        raise HTTPException(status_code=500, detail="Intervention generation failed")
