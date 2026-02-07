"""
Pydantic models for workout-related requests and responses.
Provides runtime validation and auto-documentation.
"""
from typing import Optional
from pydantic import BaseModel, Field


class WorkoutRequest(BaseModel):
    """Request model for workout generation."""
    
    fitnessLevel: str = Field(..., description="User's fitness level", examples=["beginner", "intermediate", "advanced"])
    equipment: list[str] = Field(default=[], description="Available equipment")
    timeAvailable: int = Field(..., ge=10, le=180, description="Workout duration in minutes")
    injuries: list[str] = Field(default=[], description="User's injuries or limitations")
    motivationLevel: Optional[str] = Field(None, description="Current motivation level")
    workoutTiming: Optional[str] = Field(None, description="Preferred workout time")
    goalBarriers: list[str] = Field(default=[], description="Barriers to achieving goals")
    age: Optional[int] = Field(None, ge=10, le=120)
    gender: Optional[str] = Field(None)
    weight: Optional[float] = Field(None, ge=20, le=300)
    cyclePhase: Optional[str] = Field(None, description="Menstrual cycle phase (for females)")
    memoryContext: Optional[dict] = Field(default={}, description="Memory context from Mem0")

    class Config:
        json_schema_extra = {
            "example": {
                "fitnessLevel": "intermediate",
                "equipment": ["dumbbells", "resistance bands"],
                "timeAvailable": 45,
                "injuries": [],
                "motivationLevel": "high",
                "workoutTiming": "morning"
            }
        }


class ExerciseItem(BaseModel):
    """Single exercise in a workout plan."""
    
    name: str
    sets: Optional[int] = None
    reps: Optional[str] = None
    duration: Optional[str] = None
    notes: Optional[str] = None


class WorkoutResponse(BaseModel):
    """Response model for generated workout."""
    
    title: str
    duration: str
    focus: str
    warmup: list[ExerciseItem]
    exercises: list[ExerciseItem]
    cooldown: list[ExerciseItem]


class WorkoutAPIResponse(BaseModel):
    """API wrapper response for workout endpoint."""
    
    status: str = "success"
    plan: WorkoutResponse
