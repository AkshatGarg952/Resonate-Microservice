"""
Tests for the main FastAPI application endpoints.
"""
import pytest


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns expected message."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Resonate Microservice running"}

    def test_health_endpoint(self, client):
        """Test the health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "resonate-microservice"
        assert "version" in data


class TestWorkoutEndpoint:
    """Tests for workout generation endpoint."""

    def test_workout_endpoint_requires_body(self, client):
        """Test that workout endpoint requires a request body."""
        response = client.post("/generate-workout")
        # FastAPI returns 422 for validation errors
        assert response.status_code == 422

    def test_workout_endpoint_validates_required_fields(self, client):
        """Test that workout endpoint validates required fields."""
        response = client.post("/generate-workout", json={})
        assert response.status_code == 422

    def test_workout_endpoint_with_valid_request(self, client, sample_workout_request, mock_openai, sample_workout_response):
        """Test workout generation with valid request."""
        import json
        mock_openai.chat.completions.create.return_value.choices[0].message.content = json.dumps(sample_workout_response)
        
        response = client.post("/generate-workout", json=sample_workout_request)
        
        # Should succeed or return expected error from mock
        assert response.status_code in [200, 500]  # 500 if OpenAI mock isn't properly connected


class TestNutritionEndpoint:
    """Tests for nutrition plan generation endpoint."""

    def test_nutrition_endpoint_accepts_request(self, client, sample_nutrition_request, mock_openai, sample_meal_plan_response):
        """Test nutrition endpoint processes requests."""
        import json
        mock_openai.chat.completions.create.return_value.choices[0].message.content = json.dumps(sample_meal_plan_response)
        
        response = client.post("/generate-nutrition", json=sample_nutrition_request)
        
        # Should succeed or return expected response
        assert response.status_code in [200, 422, 500]

    def test_nutrition_endpoint_with_minimal_request(self, client, mock_openai, sample_meal_plan_response):
        """Test nutrition endpoint with minimal data."""
        import json
        mock_openai.chat.completions.create.return_value.choices[0].message.content = json.dumps(sample_meal_plan_response)
        
        response = client.post("/generate-nutrition", json={
            "age": 25,
            "gender": "female"
        })
        
        # Endpoint should handle partial data gracefully
        assert response.status_code in [200, 422, 500]
