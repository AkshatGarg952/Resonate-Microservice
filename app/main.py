"""
Resonate Microservice - Main Entry Point

AI-powered diagnostics parser and fitness/nutrition generator.
"""
import os
from fastapi import FastAPI

from app.core.config import settings
from app.core.logger import logger
from app.routes import parser, workout, nutrition


# Validate configuration on startup
try:
    settings.validate()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise


# Create FastAPI app
app = FastAPI(
    title="Resonate Microservice",
    description="AI-powered diagnostics parser and fitness/nutrition generator",
    version="1.0.0"
)


# Include route modules
app.include_router(parser.router, tags=["Parser"])
app.include_router(workout.router, tags=["Workout"])
app.include_router(nutrition.router, tags=["Nutrition"])


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Resonate Microservice running"}


@app.get("/health")
def health():
    """Detailed health status."""
    return {
        "status": "healthy",
        "service": "resonate-microservice",
        "version": "1.0.0"
    }


# Run with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
