"""
Resonate Microservice - Main Entry Point

AI-powered diagnostics parser and fitness/nutrition generator.
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.logger import logger
from app.core.limiter import limiter
from app.routes import parser, workout, nutrition, intervention


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

# Attach rate limiter and its error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Include route modules — pass limiter so routes can apply per-endpoint limits
app.include_router(parser.router, tags=["Parser"])
app.include_router(workout.router, tags=["Workout"])
app.include_router(nutrition.router, tags=["Nutrition"])
app.include_router(intervention.router, tags=["Intervention"])


@app.get("/")
@limiter.limit("60/minute")
def root(request: Request):
    """Health check endpoint."""
    return {"message": "Resonate Microservice running"}


@app.get("/health")
def health():
    """
    Detailed health status.
    Returns 'degraded' if required environment variables are missing.
    """
    required_vars = ["OPENAI_API_KEY", "INTERNAL_API_SECRET"]
    missing = [v for v in required_vars if not os.environ.get(v)]

    if missing:
        return JSONResponse(
            status_code=200,
            content={
                "status": "degraded",
                "service": "resonate-microservice",
                "version": "1.0.0",
                "missing_config": missing,
                "message": f"Missing required environment variables: {', '.join(missing)}"
            }
        )

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
        reload=False  # Never use reload=True in production (disables multi-threading)
    )
