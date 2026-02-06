"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel


# --- Parser Models ---

class ParseRequest(BaseModel):
    """Request model for blood report parsing."""
    pdfUrl: str
    biomarkers: list[str]


class ParseResponse(BaseModel):
    """Response model for parsed biomarkers."""
    confidence: str
    totalBiomarkers: int
    foundBiomarkers: int
    missingBiomarkers: list[str]
    values: dict[str, float | None]


# --- Workout Models ---

class WorkoutRequest(BaseModel):
    """Request model for workout generation."""
    fitnessLevel: str
    equipment: list[str]
    timeAvailable: int
    injuries: list[str]
    motivationLevel: str | None = None
    workoutTiming: str | None = None
    goalBarriers: list[str] = []
    age: int | None = None
    gender: str | None = None
    weight: float | None = None
    cyclePhase: str | None = None


class ExerciseItem(BaseModel):
    """Single exercise in a workout."""
    name: str
    sets: int | None = None
    reps: str | None = None
    duration: str | None = None
    notes: str | None = None


class WorkoutPlan(BaseModel):
    """Generated workout plan."""
    title: str
    duration: str
    focus: str
    warmup: list[ExerciseItem]
    exercises: list[ExerciseItem]
    cooldown: list[ExerciseItem]


# --- Nutrition Models ---

class NutritionRequest(BaseModel):
    """Request model for nutrition plan generation."""
    age: int | None = None
    gender: str | None = None
    weight: float | None = None
    height: float | None = None
    goals: str | None = None
    dietType: str | None = None
    allergies: list[str] = []
    cuisine: str = "Indian"


class MealItem(BaseModel):
    """Single meal in a nutrition plan."""
    name: str
    description: str
    calories: int
    protein: str


class NutritionPlan(BaseModel):
    """Generated nutrition plan."""
    breakfast: MealItem
    lunch: MealItem
    dinner: MealItem
    snacks: list[MealItem]
    total_calories: int
    total_protein: str


# --- Food Analysis Models ---

class FoodAnalysisRequest(BaseModel):
    """Request model for food image analysis."""
    imageUrl: str
    cuisine: str = "General"


class NutritionalInfo(BaseModel):
    """Nutritional breakdown of food."""
    calories: int
    protein: str
    carbohydrates: str
    fats: str
    fiber: str


class FoodAnalysis(BaseModel):
    """Food analysis result."""
    food_name: str
    description: str
    ingredients: list[str]
    nutritional_info: NutritionalInfo
    health_rating: str
    suggestions: str


# --- Generic Response Models ---

class SuccessResponse(BaseModel):
    """Generic success response wrapper."""
    status: str = "success"
    plan: dict | None = None
    analysis: dict | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
