"""
Pytest fixtures for the Resonate Microservice tests.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Mock environment variables before importing app
import os
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_openai():
    """Mock OpenAI API calls."""
    with patch("app.services.openai_service.client") as mock:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"test": "response"}'))
        ]
        mock.chat.completions.create.return_value = mock_response
        yield mock


@pytest.fixture
def sample_workout_request():
    """Sample workout generation request."""
    return {
        "fitnessLevel": "intermediate",
        "equipment": ["dumbbells", "barbell"],
        "timeAvailable": 45,
        "injuries": [],
        "motivationLevel": "high",
        "workoutTiming": "morning",
        "goalBarriers": []
    }


@pytest.fixture
def sample_nutrition_request():
    """Sample nutrition plan request."""
    return {
        "age": 30,
        "gender": "male",
        "weight": 75,
        "height": 175,
        "goals": "muscle gain",
        "dietType": "vegetarian",
        "allergies": ["peanuts"],
        "cuisine": "Indian"
    }


@pytest.fixture
def sample_workout_response():
    """Sample workout plan response."""
    return {
        "title": "Morning Power Session",
        "duration": "45 Minutes",
        "focus": "Full Body Strength",
        "warmup": [
            {"name": "Jumping Jacks", "duration": "2 mins"}
        ],
        "exercises": [
            {"name": "Squats", "sets": 4, "reps": "10-12", "notes": "Keep core tight"}
        ],
        "cooldown": [
            {"name": "Stretching", "duration": "5 mins"}
        ]
    }


@pytest.fixture
def sample_meal_plan_response():
    """Sample meal plan response."""
    return {
        "breakfast": {
            "name": "Oatmeal with Fruits",
            "description": "Steel-cut oats with banana and berries",
            "calories": 350,
            "protein": "12g"
        },
        "lunch": {
            "name": "Dal Tadka with Rice",
            "description": "Yellow lentils with tempered spices",
            "calories": 450,
            "protein": "15g"
        },
        "dinner": {
            "name": "Paneer Tikka",
            "description": "Grilled cottage cheese with bell peppers",
            "calories": 400,
            "protein": "20g"
        },
        "snacks": [
            {"name": "Fruit Bowl", "description": "Mixed fruits", "calories": 150, "protein": "2g"}
        ],
        "total_calories": 1350,
        "total_protein": "49g"
    }
