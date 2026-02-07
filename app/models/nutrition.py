"""
Pydantic models for nutrition-related requests and responses.
"""
from typing import Optional
from pydantic import BaseModel, Field


class NutritionRequest(BaseModel):
    """Request model for nutrition/meal plan generation."""
    
    age: Optional[int] = Field(None, ge=10, le=120)
    gender: Optional[str] = Field(None)
    weight: Optional[float] = Field(None, ge=20, le=300, description="Weight in kg")
    height: Optional[float] = Field(None, ge=50, le=300, description="Height in cm")
    goals: Optional[str] = Field(None, max_length=500)
    dietType: Optional[str] = Field(None, description="e.g., vegetarian, non-vegetarian")
    allergies: list[str] = Field(default=[], description="Food allergies or restrictions")
    cuisine: str = Field(default="Indian", description="Preferred cuisine type")
    memoryContext: Optional[dict] = Field(default={}, description="Memory context from Mem0")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 28,
                "gender": "female",
                "weight": 60,
                "height": 165,
                "goals": "weight loss",
                "dietType": "vegetarian",
                "allergies": ["peanuts"],
                "cuisine": "Indian"
            }
        }


class MealItem(BaseModel):
    """Single meal in the plan."""
    
    name: str
    description: str
    calories: int
    protein: str
    carbs: Optional[str] = None
    fats: Optional[str] = None


class NutritionResponse(BaseModel):
    """Response model for generated meal plan."""
    
    breakfast: MealItem
    lunch: MealItem
    dinner: MealItem
    snacks: list[MealItem] = []
    total_calories: int
    total_protein: str
    total_carbs: Optional[str] = None
    total_fats: Optional[str] = None


class NutritionAPIResponse(BaseModel):
    """API wrapper response for nutrition endpoint."""
    
    status: str = "success"
    plan: NutritionResponse
