"""
Workout generation routes.
"""
from fastapi import APIRouter, HTTPException, Depends, Request

from app.models.schemas import WorkoutRequest
from app.services import openai_service
from app.core.logger import log_request, log_error
from app.core.auth import verify_internal_secret
from app.core.limiter import limiter

router = APIRouter(dependencies=[Depends(verify_internal_secret)])


@router.post("/generate-workout")
@limiter.limit("10/minute")
async def generate_workout(request: Request, req: WorkoutRequest):
    """
    Generate personalized AI workout plan.

    Takes user profile (fitness level, equipment, injuries, etc.)
    and generates a customized workout routine.
    """
    log_request("/generate-workout")

    try:
        plan = await openai_service.generate_workout(
            level=req.fitnessLevel,
            equipment=req.equipment,
            time=req.timeAvailable,
            injuries=req.injuries,
            motivation=req.motivationLevel,
            timing=req.workoutTiming,
            barriers=req.goalBarriers,
            age=req.age,
            gender=req.gender,
            weight=req.weight,
            cycle_phase=req.cyclePhase
        )
        return {"status": "success", "plan": plan}
    except ValueError as e:
        log_error("Workout generation", e)
        raise HTTPException(status_code=500, detail="AI generation failed to produce valid JSON")
    except Exception as e:
        log_error("Workout generation", e)
        raise HTTPException(status_code=500, detail="Workout generation failed")
