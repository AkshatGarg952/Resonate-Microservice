"""
Tests for the OpenAI service functions.
"""
import pytest
from unittest.mock import patch, MagicMock
import json


class TestSanitizeKey:
    """Tests for the sanitize_key utility function."""

    def test_converts_to_camel_case(self):
        """Test that biomarker names are converted to camelCase."""
        from app.services.openai_service import sanitize_key
        
        assert sanitize_key("Hemoglobin") == "hemoglobin"
        assert sanitize_key("Red Blood Cell Count") == "redBloodCellCount"
        assert sanitize_key("WHITE BLOOD CELLS") == "whiteBloodCells"

    def test_removes_special_characters(self):
        """Test that special characters are removed."""
        from app.services.openai_service import sanitize_key
        
        assert sanitize_key("Vitamin B-12") == "vitaminB12"
        assert sanitize_key("T3 (Total)") == "t3Total"

    def test_handles_empty_string(self):
        """Test handling of empty or whitespace strings."""
        from app.services.openai_service import sanitize_key
        
        assert sanitize_key("") == "biomarker"
        assert sanitize_key("   ") == "biomarker"


class TestCallChatAPI:
    """Tests for the call_chat_api function."""

    @patch("app.services.openai_service.client")
    def test_successful_api_call(self, mock_client):
        """Test successful chat API call returns parsed JSON."""
        from app.services.openai_service import call_chat_api
        
        expected_response = {"workout": {"title": "Test Workout"}}
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=json.dumps(expected_response)))
        ]
        
        result = call_chat_api("System prompt", "User prompt")
        
        assert result == expected_response
        mock_client.chat.completions.create.assert_called_once()

    @patch("app.services.openai_service.client")
    def test_invalid_json_response_raises_error(self, mock_client):
        """Test that invalid JSON response raises ValueError."""
        from app.services.openai_service import call_chat_api
        
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Not valid JSON"))
        ]
        
        with pytest.raises(ValueError, match="AI did not return valid JSON"):
            call_chat_api("System prompt", "User prompt")


class TestGenerateWorkout:
    """Tests for the generate_workout function."""

    @patch("app.services.openai_service.call_chat_api")
    def test_generates_workout_with_basic_params(self, mock_call_api):
        """Test workout generation with basic parameters."""
        from app.services.openai_service import generate_workout
        
        expected = {"title": "Morning Workout", "exercises": []}
        mock_call_api.return_value = expected
        
        result = generate_workout(
            level="beginner",
            equipment=["dumbbells"],
            time=30,
            injuries=[]
        )
        
        assert result == expected
        mock_call_api.assert_called_once()

    @patch("app.services.openai_service.call_chat_api")
    def test_includes_optional_params_in_prompt(self, mock_call_api):
        """Test that optional parameters are included in the prompt."""
        from app.services.openai_service import generate_workout
        
        mock_call_api.return_value = {}
        
        generate_workout(
            level="advanced",
            equipment=["barbell", "dumbbells"],
            time=60,
            injuries=["lower back"],
            motivation="low",
            timing="evening",
            barriers=["time constraints"],
            age=35,
            gender="female",
            weight=65.0,
            cycle_phase="luteal"
        )
        
        # Check that the user prompt includes the optional parameters
        call_args = mock_call_api.call_args
        user_prompt = call_args[0][1]  # Second positional argument
        
        assert "35" in user_prompt  # age
        assert "female" in user_prompt  # gender
        assert "65" in user_prompt  # weight
        assert "lower back" in user_prompt  # injury


class TestGenerateMealPlan:
    """Tests for the generate_meal_plan function."""

    @patch("app.services.openai_service.call_chat_api")
    def test_generates_meal_plan(self, mock_call_api):
        """Test meal plan generation."""
        from app.services.openai_service import generate_meal_plan
        
        expected = {
            "breakfast": {"name": "Oatmeal"},
            "lunch": {"name": "Salad"},
            "dinner": {"name": "Grilled fish"},
            "total_calories": 1500
        }
        mock_call_api.return_value = expected
        
        result = generate_meal_plan(
            age=30,
            gender="male",
            weight=80,
            height=180,
            goals="weight loss",
            diet_type="vegetarian"
        )
        
        assert result == expected

    @patch("app.services.openai_service.call_chat_api")
    def test_handles_allergies(self, mock_call_api):
        """Test that allergies are included in the prompt."""
        from app.services.openai_service import generate_meal_plan
        
        mock_call_api.return_value = {}
        
        generate_meal_plan(
            allergies=["peanuts", "shellfish"],
            cuisine="Indian"
        )
        
        call_args = mock_call_api.call_args
        user_prompt = call_args[0][1]
        
        assert "peanuts" in user_prompt
        assert "shellfish" in user_prompt
        assert "Indian" in user_prompt
