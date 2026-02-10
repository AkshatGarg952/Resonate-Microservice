"""
Nutrition and food analysis routes.
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import FoodAnalysisRequest
from app.models.nutrition import NutritionRequest
from app.services import pdf_service, openai_service
from app.core.logger import log_request, log_error

router = APIRouter()


@router.post("/generate-nutrition")
def generate_nutrition(req: NutritionRequest):
    """
    Generate personalized daily meal plan.
    
    Creates breakfast, lunch, dinner, and snacks based on
    user profile, dietary preferences, and goals.
    """
    log_request("/generate-nutrition")
    
    try:
        plan = openai_service.generate_meal_plan(
            age=req.age,
            gender=req.gender,
            weight=req.weight,
            height=req.height,
            goals=req.goals,
            diet_type=req.dietType,
            allergies=req.allergies,
            cuisine=req.cuisine
        )
        return {"status": "success", "plan": plan}
    except ValueError as e:
        log_error("Nutrition generation", e)
        raise HTTPException(status_code=500, detail="AI generation failed to produce valid JSON")
    except Exception as e:
        log_error("Nutrition generation", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-food")
def analyze_food(req: FoodAnalysisRequest):
    """
    Analyze food image for nutritional content.
    
    Downloads image, sends to AI for analysis,
    returns nutritional breakdown and health rating.
    """
    log_request("/analyze-food")
    
    # Download image
    try:
        image_bytes = pdf_service.download_file(req.imageUrl)
        image_base64 = pdf_service.image_to_base64(image_bytes)
    except Exception as e:
        log_error("Image download", e)
        raise HTTPException(status_code=400, detail="Could not download or process image")

    # Analyze with AI
    try:
        analysis = openai_service.analyze_food_image(image_base64, req.cuisine)
        return {"status": "success", "analysis": analysis}
    except ValueError as e:
        log_error("Food analysis", e)
        raise HTTPException(status_code=500, detail="AI generation failed to produce valid JSON")
    except Exception as e:
        log_error("Food analysis", e)
        raise HTTPException(status_code=500, detail=str(e))
